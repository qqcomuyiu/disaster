import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from tqdm import tqdm

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # 1. 读取数据
    df = pd.read_csv('train.csv')

    # 2. 文本清理函数
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r" \1", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df['clean_text'] = df['text'].apply(clean_text)

    # 3. 数据处理
    train_texts = df['clean_text'].tolist()
    train_labels = df['target'].tolist()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def encode_texts(text_list, max_len=128):
        input_ids, attention_masks = [], []
        for t in text_list:
            encoded = tokenizer.encode_plus(
                t,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

    input_ids, attention_masks = encode_texts(train_texts)
    labels = torch.tensor(train_labels)

    # 4. 划分训练集和验证集
    train_ids, val_ids, train_masks, val_masks, train_y, val_y = train_test_split(
        input_ids, attention_masks, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # 5. 数据加载器
    train_data = TensorDataset(train_ids, train_masks, train_y)
    val_data = TensorDataset(val_ids, val_masks, val_y)

    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), num_workers=4, batch_size=32)
    val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), num_workers=4, batch_size=32)

    # 6. 模型初始化
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 7. 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 3
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 8. 加权交叉熵
    # 注意：这里的class_counts和权重计算不太典型，通常需要根据数据集类别分布计算真实权重
    # 这里按照你之前的写法进行，确保为float32
    class_counts = np.array([float(1/3473), float(1/2617)], dtype=np.float32)
    class_weights = torch.tensor(class_counts, dtype=torch.float32).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 9. 训练与验证循环
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # 训练阶段
        model.train()
        total_train_loss = 0
        train_preds, train_labels_true = [], []

        for batch in tqdm(train_dataloader, desc="Training"):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            # logits通常已经是float32，不需要转换
            loss = loss_function(logits, b_labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 计算训练集预测结果
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_labels_true.extend(b_labels.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = accuracy_score(train_labels_true, train_preds)
        print(f"  Average training loss: {avg_train_loss:.4f}")
        print(f"  Training Accuracy: {train_accuracy:.4f}")

        # 验证阶段
        model.eval()
        all_logits, all_labels = [], []

        for batch in tqdm(val_dataloader, desc="Validation"):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs.logits.cpu().numpy()
                all_logits.extend(logits)
                all_labels.extend(b_labels.cpu().numpy())

        # 动态阈值调整
        probs = torch.softmax(torch.tensor(all_logits), dim=1)[:, 1].numpy()
        precision, recall, thresholds = precision_recall_curve(all_labels, probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]

        print(f"  Best F1 Score: {best_f1:.4f}, Best Threshold: {best_threshold:.4f}")

        # 使用最佳阈值计算最终预测结果
        final_preds = (probs > best_threshold).astype(int)
        final_f1 = f1_score(all_labels, final_preds)
        val_accuracy = accuracy_score(all_labels, final_preds)

        print(f"  Final Validation F1 Score: {final_f1:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")



    print("Predicting on test set...")
    
    # 读取测试集数据
    test_df = pd.read_csv('test.csv')  # 假设测试集数据位于 test.csv 文件中
    test_texts = test_df['text'].tolist()
    test_ids = test_df['id'].tolist()  # 测试集的 ID 列
    
    # 编码测试集文本
    test_input_ids, test_attention_masks = encode_texts(test_texts)
    
    # 创建测试集数据加载器
    test_data = TensorDataset(test_input_ids, test_attention_masks)
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=32)
    
    # 设置模型为评估模式
    model.eval()
    test_preds = []
    
    for batch in tqdm(test_dataloader, desc="Testing"):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
    
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]  # 获取正类概率
            preds = (probs > best_threshold).long()  # 使用最佳阈值进行分类
            test_preds.extend(preds.cpu().numpy())
    
    # 将预测结果与 ID 结合
    test_results = pd.DataFrame({'id': test_ids, 'target': test_preds})
    
    # 保存为 CSV 文件
    test_results.to_csv('submission.csv', index=False)
    print("Test predictions saved to submission.csv.")
