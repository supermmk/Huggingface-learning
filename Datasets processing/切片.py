# ============================================================
# 分割和整理数据 · 完整流程
# 数据集：药物评论数据集（UCI 机器学习仓库）
# ============================================================

# ---------- 0. 下载数据（命令行） ----------
# !wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
# !unzip drugsCom_raw.zip


# ============================================================
# 1. 加载 TSV 数据
# ============================================================
from datasets import load_dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
#                                                    ↑ TSV 用制表符分隔


# ============================================================
# 2. 探索数据：随机抽样
# ============================================================
# 先打乱，再取前 1000 条（seed 固定，可复现）
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
print(drug_sample[:3])


# ============================================================
# 3. 验证猜想：Unnamed: 0 是不是患者 ID
# ============================================================
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))
# 断言通过 → 每行 ID 唯一 → 确认是患者 ID


# ============================================================
# 4. 重命名列
# ============================================================
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0",
    new_column_name="patient_id",
)
# DatasetDict 的方法，自动应用到所有 split


# ============================================================
# 5. 清洗 condition 列
# ============================================================
# 5.1 先定义规范化函数
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

# 5.2 直接 map 会报错：'NoneType' object has no attribute 'lower'
# drug_dataset.map(lowercase_condition)   # ❌

# 5.3 先过滤掉 None
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

# 5.4 再规范化
drug_dataset = drug_dataset.map(lowercase_condition)
print(drug_dataset["train"]["condition"][:3])
# ['left ventricular dysfunction', 'adhd', 'birth control']


# ============================================================
# 6. 创建新列：review_length
# ============================================================
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}
    # 返回的 key 不在原数据集里 → 自动创建新列

drug_dataset = drug_dataset.map(compute_review_length)
print(drug_dataset["train"][0])
# 多了 'review_length' 字段


# ============================================================
# 7. 查看极端情况：排序
# ============================================================
print(drug_dataset["train"].sort("review_length")[:3])
# 发现有些评论只有 1 个词


# ============================================================
# 8. 过滤过短的评论
# ============================================================
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)
# {'train': 138514, 'test': 46108}   ← 删了约 15%


# ============================================================
# 9. 解码 HTML 字符
# ============================================================
import html

# 9.1 逐条处理
drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})

# 9.2 批量处理（快）——用列表推导
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]},
    batched=True,
)


# ============================================================
# 10. 使用快速 tokenizer 加速
# ============================================================
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")   # 默认 use_fast=True

def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)

# 批量处理
tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)

# 慢速 tokenizer + 多进程（可选）
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)

def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)

# tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)


# ============================================================
# 11. 处理变长：return_overflowing_tokens
# ============================================================
def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,   # 长文本切成多段
    )

# 11.1 先在单条上测试
result = tokenize_and_split(drug_dataset["train"][0])
print([len(inp) for inp in result["input_ids"]])   # [128, 49]  → 切成 2 段

# 11.2 直接 map 会报错：列长度不一致（1463 vs 1000）
# tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)   # ❌

# 11.3 方案 1：删掉原列
tokenized_dataset = drug_dataset.map(
    tokenize_and_split,
    batched=True,
    remove_columns=drug_dataset["train"].column_names,
)
print(len(tokenized_dataset["train"]), len(drug_dataset["train"]))
# (206772, 138514)   ← 新数据集行数更多

# 11.4 方案 2：用 overflow_to_sample_mapping 保留原列
def tokenize_and_split_keep(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # 提取"新片段 → 原样本"的映射
    sample_map = result.pop("overflow_to_sample_mapping")
    # 按映射复制原列的值，使长度一致
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result

tokenized_dataset_keep = drug_dataset.map(tokenize_and_split_keep, batched=True)


# ============================================================
# 12. 与 Pandas 互转，存储模式不变，只改变取出形式
# ============================================================
# 12.1 切换到 pandas 格式
drug_dataset.set_format("pandas")

# 12.2 拿 DataFrame（必须用 [:]）
train_df = drug_dataset["train"][:]
print(type(train_df))    # pandas.DataFrame
print(type(drug_dataset["train"]))   # 还是 Dataset！

# 12.3 用 Pandas 分析：condition 分布
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "count": "frequency"})
)
print(frequencies.head())

# 12.4 转回 Dataset
from datasets import Dataset
freq_dataset = Dataset.from_pandas(frequencies)
print(freq_dataset)

# 12.5 重置格式
drug_dataset.reset_format()


# ============================================================
# 13. 划分验证集
# ============================================================
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# 把默认的 "test" 改名为 "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# 把原始 test 加进来
drug_dataset_clean["test"] = drug_dataset["test"]

print(drug_dataset_clean)
# DatasetDict({train, validation, test})


# ============================================================
# 14. 保存数据集
# ============================================================
# 14.1 Arrow 格式（推荐）
drug_dataset_clean.save_to_disk("drug-reviews")

# 14.2 从磁盘加载
from datasets import load_from_disk
drug_dataset_reloaded = load_from_disk("drug-reviews")
print(drug_dataset_reloaded)

# 14.3 JSON 格式（分 split 存）
for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")

# 14.4 从 JSON 加载
data_files = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews-test.jsonl",
}
drug_dataset_reloaded = load_dataset("json", data_files=data_files)