# ============================================================
# 从头训练因果语言模型（GPT-2 风格）
# 任务：代码自动补全
# 数据：codeparrot 流式过滤（含 pandas/sklearn/matplotlib/seaborn）
# 保存：本地
# ============================================================

import torch
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


# ============================================================
# 1. 收集数据（流式过滤）
# ============================================================
def any_keyword_in_string(string, keywords):
    """检查字符串是否包含任一关键词"""
    for keyword in keywords:
        if keyword in string:
            return True
    return False


def filter_streaming_dataset(dataset, filters):
    """
    从流式数据集中筛出"含关键词"的样本。

    参数:
        dataset: 流式数据集（streaming=True 加载的 IterableDataset）
        filters: 关键词列表

    返回:
        Dataset（普通，筛选后的数据）

    为什么不用 .filter()？
        流式数据集不支持 .filter()（不知道全貌），只能手动遍历。
    """
    # 用 defaultdict(list) 存结果
    #   - key = 列名
    #   - value = 该列所有值的列表
    #   - 访问不存在的 key 时自动创建空列表
    filtered_dict = defaultdict(list)

    total = 0  # 总样本数（用于算筛选比例）

    # 遍历流式数据集
    #   iter(dataset)  把 IterableDataset 转成迭代器
    #   tqdm(...)      加进度条
    #   sample         一条数据（dict）
    for sample in tqdm(iter(dataset)):
        total += 1  # 计数 +1

        # 检查这条样本的 content 是否含任一关键词
        if any_keyword_in_string(sample["content"], filters):
            # 含关键词 → 保留这条样本的所有字段
            #   sample.items()  遍历 (列名, 值)
            for k, v in sample.items():
                filtered_dict[k].append(v)  # 按列收集

    # 打印筛选比例（筛出数 / 总数）
    #   {:.2%}  格式化为百分比（如 3.45%）
    print(f"{len(filtered_dict['content']) / total:.2%} of data after filtering.")

    # 把"列式字典"转成 Dataset
    #   filtered_dict = {"content": [...], "path": [...], ...}
    #   → Dataset（表格）
    return Dataset.from_dict(filtered_dict)


# 关键词：数据科学相关库
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

# 流式加载 + 过滤（train / valid 分别处理）
print("=== 过滤 train ===")
data_train = load_dataset(
    "transformersbook/codeparrot-train",
    split="train",
    streaming=True,                 # ← 流式，不下载全部  streaming: 不下载，边用边取（像"在线看视频" vs "先下载再看"）
)
filtered_train = filter_streaming_dataset(data_train, filters)

print("=== 过滤 valid ===")
data_valid = load_dataset(
    "transformersbook/codeparrot-valid",
    split="valid",
    streaming=True,
)
filtered_valid = filter_streaming_dataset(data_valid, filters)

# 组装成 DatasetDict
raw_datasets = DatasetDict({
    "train": filtered_train,
    "valid": filtered_valid,
})

print(raw_datasets)

# 保存过滤后的数据（避免每次重新过滤）
raw_datasets.save_to_disk("codeparrot-ds-filtered")
# 之后可以直接：
# from datasets import load_from_disk
# raw_datasets = load_from_disk("codeparrot-ds-filtered")


# ============================================================
# 2. 加载 tokenizer
# ============================================================
context_length = 128

tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

# GPT-2 默认无 pad_token，用 eos_token 替代
tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 3. 数据处理：切块成 128 token
# ============================================================
def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,          # 128
        return_overflowing_tokens=True,     # 超长切成多块
        return_length=True,
    )

    # 只保留"长度 == 128"的完整块
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)

    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,   # 行数变了，删原列
)

print(tokenized_datasets)


# ============================================================
# 4. 配置模型（从零初始化 GPT-2）
# ============================================================
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),              # 词表大小 = tokenizer 大小
    n_ctx=context_length,                   # 上下文长度 = 128
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)             # 权重随机初始化

model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")


# ============================================================
# 5. DataCollator（语言模型专用）
# ============================================================
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# ============================================================
# 6. 配置训练参数（本地保存）
# ============================================================
# ============================================================
# 6. 配置训练参数（本地保存）
# ============================================================
args = TrainingArguments(
    # ---- 保存 ----
    output_dir="codeparrot-ds",  # 输出目录（checkpoint、日志、模型都存这）

    # ---- batch 大小 ----
    per_device_train_batch_size=32,  # 每个设备（GPU）训练 batch 大小
    per_device_eval_batch_size=32,  # 每个设备评估 batch 大小
    # 有效 batch = per_device × grad_accum × 卡数 = 32 × 8 × 1 = 256

    # ---- 评估 ----
    eval_strategy="steps",  # 评估策略：按步评估
    # ⚠️ 新版参数名（旧版叫 evaluation_strategy）
    # 可选："no"（不评估）、"steps"（按步）、"epoch"（按轮）
    eval_steps=5_000,  # 每 5000 步评估一次

    # ---- 日志 ----
    logging_steps=5_000,  # 每 5000 步打印一次训练日志（loss 等）

    # ---- 梯度累积 ----
    gradient_accumulation_steps=8,  # 梯度累积 8 次才更新一次参数
    # 作用：用 32 的显存，达到 256 的 batch 效果
    # 有效 batch = 32 × 8 = 256

    # ---- 训练轮数 ----
    num_train_epochs=1,  # 训练 1 轮（整个数据集过 1 遍）

    # ---- 优化 ----
    weight_decay=0.1,  # 权重衰减（正则化，防过拟合）迫使模型倾向于学习较小的权重值，从而使模型函数更加平滑，对输入的微小扰动不那么敏感。可以问ai要公式
    learning_rate=5e-4,  # 学习率（5 × 10^-4）

    # ---- 学习率调度 ----
    warmup_steps=1_000,  # 前 1000 步预热（学习率从 0 升到设定值）
    lr_scheduler_type="cosine",  # 调度策略：余弦衰减
    # 可选："linear"、"cosine"、"constant" 等

    # ---- 保存 ----
    save_steps=5_000,  # 每 5000 步保存一个 checkpoint
    save_total_limit=3,  # 最多保留 3 个 checkpoint（旧的自动删）
    # 防止 checkpoint 占满磁盘

    # ---- 性能 ----
    fp16=torch.cuda.is_available(),  # 混合精度：有 GPU 才开
    # fp16=True → 用 float16，省显存、加速
    # fp16=False → 用 float32（CPU 上必须）

    # ---- Hub ----
    push_to_hub=False,  # 不上传 Hub，只保存本地
)

# ============================================================
# 7. 组装 Trainer
# ============================================================
trainer = Trainer(
    model=model,  # 要训练的模型
    args=args,  # 训练配置（上面那个）
    data_collator=data_collator,  # 怎么组 batch（语言模型专用）
    train_dataset=tokenized_datasets["train"],  # 训练集
    eval_dataset=tokenized_datasets["valid"],  # 验证集
    processing_class=tokenizer,  # tokenizer
    # ⚠️ 新版参数名（旧版叫 tokenizer=...）
    # 用于：保存时一起存 tokenizer、日志里显示
)


# ============================================================
# 8. 训练
# ============================================================
trainer.train()


# ============================================================
# 9. 保存最终模型到本地
# ============================================================
trainer.save_model("codeparrot-ds-final")
tokenizer.save_pretrained("codeparrot-ds-final")


# ============================================================
# 10. 之后加载（验证）
# ============================================================
from transformers import AutoModelForCausalLM, AutoTokenizer

loaded_model = AutoModelForCausalLM.from_pretrained("codeparrot-ds-final")
loaded_tokenizer = AutoTokenizer.from_pretrained("codeparrot-ds-final")

inputs = loaded_tokenizer("import pandas as", return_tensors="pt")
with torch.no_grad():
    outputs = loaded_model.generate(**inputs, max_new_tokens=50)
print(loaded_tokenizer.decode(outputs[0], skip_special_tokens=True))