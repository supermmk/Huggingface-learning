# ============================================================
# 不用 Trainer，手写训练 + 评估完整流程
# 任务：MRPC 句子对分类
# ============================================================

# ---------- 1. 导入 ----------
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler,
)


# ============================================================
# 2. 数据预处理（和第 2 节一样）
# ============================================================
raw_datasets = load_dataset("nyu-mll/glue", "mrpc")   # ⚠️ 新版名字
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
    )

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ============================================================
# 3. 后处理（自己实现 Trainer 自动做的事）
# ============================================================
# 删掉模型不需要的列
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
# 重命名 label → labels（模型/Trainer 认这个名字）
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# 转成 PyTorch 张量
tokenized_datasets.set_format("torch")

# 检查列名
print(tokenized_datasets["train"].column_names)
# ['attention_mask', 'input_ids', 'labels', 'token_type_ids']


# ============================================================
# 4. DataLoader
# ============================================================
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,              # 训练集打乱
    batch_size=8,
    collate_fn=data_collator,  # 动态 padding
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8,
    collate_fn=data_collator,
)

# 快速验证一个 batch 的形状
for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})
# {'attention_mask': [8, 65], 'input_ids': [8, 65], 'labels': [8], 'token_type_ids': [8, 65]}


# ============================================================
# 5. 加载模型
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# 验证前向传播
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
# tensor(0.5441, grad_fn=<NllLossBackward>) torch.Size([8, 2])


# ============================================================
# 6. 优化器 + 调度器
# ============================================================
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",                      # 线性衰减
    optimizer=optimizer,
    num_warmup_steps=0,            # 不预热
    num_training_steps=num_training_steps,
)
print(num_training_steps)          # 1377


# ============================================================
# 7. 设备
# ============================================================
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)


# ============================================================
# 8. 训练循环
# ============================================================
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 数据搬到设备
        batch = {k: v.to(device) for k, v in batch.items()}

        # 前向
        outputs = model(**batch)
        loss = outputs.loss

        # 反向
        loss.backward()

        # 更新参数 + 学习率
        optimizer.step()
        lr_scheduler.step()

        # 清梯度
        optimizer.zero_grad()

        progress_bar.update(1)


# ============================================================
# 9. 评估循环
# ============================================================
metric = evaluate.load("nyu-mll/glue", "mrpc")   # ⚠️ 新版名字

model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():                 # 评估不需要梯度
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())
# {'accuracy': ..., 'f1': ...}