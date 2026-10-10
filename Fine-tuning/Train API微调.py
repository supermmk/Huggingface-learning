from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================
# 5. 列重命名 + 转 torch 格式
# ============================================================
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")



# ============================================================
# 6. 加载模型
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
)

# ============================================================
# 7. 训练配置
# ============================================================
training_args = TrainingArguments(
    output_dir="test-trainer",
    eval_strategy="epoch",          # ⚠️ 新版参数名（旧版 evaluation_strategy）
)


# ============================================================
# 8. 评估指标 + compute_metrics,因为这个公开数据集有label
# ============================================================
metric = evaluate.load("nyu-mll/glue", "mrpc")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



# ============================================================
# 9. 组装 Trainer
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,     # ⚠️ 新版参数名（旧版 tokenizer）
    compute_metrics=compute_metrics,
)


# ============================================================
# 10. 开跑
# ============================================================
trainer.train()


# ============================================================
# 11. 预测 + 评估
# ============================================================
predictions = trainer.predict(tokenized_datasets["validation"])

print("logits shape:", predictions.predictions.shape)   # (408, 2)
print("labels shape:", predictions.label_ids.shape)     # (408,)
print("metrics:", predictions.metrics)                  # 含 test_loss / test_accuracy / test_f1

preds = np.argmax(predictions.predictions, axis=-1)
result = metric.compute(predictions=preds, references=predictions.label_ids)
print(result)
# {'accuracy': ..., 'f1': ...}