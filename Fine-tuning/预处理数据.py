# ============================================================
# 预处理数据完整流程
# 目标：把"人类文本"变成"模型能吃的张量 batch"
# ============================================================

# ---------- 1. 导入 ----------
from datasets import load_dataset                              # 加载数据集
from transformers import AutoTokenizer, DataCollatorWithPadding # 分词器 + 动态padding


# ---------- 2. 加载数据集 ----------
# 加载 GLUE 里的 MRPC 任务（判断两句话是否同义），这里是官方公开数据集，不一定用这个，每个处理方式不同
# ⚠️ 注意：新版要用 "nyu-mll/glue"，旧写法 "glue" 会报 HfUriError
raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
#                           ↑ 数据集名（要带命名空间）  ↑ 子集名

# 结构长这样：
# DatasetDict({
#     train:      Dataset({features: ['sentence1','sentence2','label','idx'], num_rows: 3668})
#     validation: Dataset({...})
#     test:       Dataset({...})
# })


# ---------- 3. 加载 tokenizer ----------
checkpoint = "bert-base-uncased"                               # 模型名（也是 tokenizer 名）
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# AutoTokenizer 会读 checkpoint 的 config，自动选对 tokenizer 类
# bert-base-uncased → 实际得到 BertTokenizerFast


# ---------- 4. 定义预处理函数 ----------
# ⚠️ 因为下面 map 用 batched=True，这个函数收到的是"一批"数据
#    所以参数叫 examples（复数），每个字段是 list
def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],   # 第一句的列表
        examples["sentence2"],   # 第二句的列表（句子对！）
        truncation=True,         # 超长截断到 max_length
        # 注意：这里故意不写 padding
        # 因为 padding 留给 DataCollator 在训练时动态做（更省）
    )
    # 返回 BatchEncoding：
    # {
    #   'input_ids':      [[...], [...], ...],
    #   'token_type_ids': [[...], [...], ...],   ← 句子对才有
    #   'attention_mask': [[...], [...], ...],
    # }


# ---------- 5. 用 map 批量预处理 ----------
tokenized_datasets = raw_datasets.map(
    tokenize_function,                          # 要应用的函数
    batched=True,                               # 批量处理（函数收到一批，快）
    remove_columns=["sentence1", "sentence2", "idx"],  # 删掉原列，省空间
)
# map 内部按 batch_size=1000 分批调用 tokenize_function
# 结果以 Apache Arrow 格式存磁盘，内存只放当前需要的部分（省内存）
# 处理完列变成：['input_ids', 'token_type_ids', 'attention_mask', 'label']


# ---------- 6. 重命名标签列 ----------
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# ⚠️ 必须改成 labels（复数）
# 因为模型 forward() 的参数名是 labels，Trainer 也靠这个名字找标签


# ---------- 7. 转成 PyTorch 张量 ----------
tokenized_datasets.set_format("torch")
# 把数据从 Python list 转成 torch.Tensor
# 之后取出来就是 tensor([...])，可以直接喂模型


# ---------- 8. 创建动态 padding 的 DataCollator ----------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# 它做什么：
#   1. 把一批样本堆叠成 batch → [batch, seq]
#   2. 动态 padding 到"本批最长"（不是全数据集最长，省计算）
#   3. 自动生成正确的 attention_mask（padding 位置为 0）


# ============================================================
# 数据流总结
# ============================================================
#
# 原始数据（文本+标签）
#     ↓ load_dataset
# DatasetDict
#     ↓ map(tokenize_function, batched=True)   ← Arrow 磁盘，省内存
# Tokenized Dataset（变长，未 padding）
#     ↓ rename label→labels, set_format("torch")
#     ↓ DataCollatorWithPadding（训练时按 batch 动态 padding）
# Batch（矩形张量 [batch, seq]）
#     ↓
# model(**batch)
#
# ============================================================


# ============================================================
# 关键点速查
# ============================================================
#
# 1. 句子对：tokenizer(句子1列表, 句子2列表) 两个位置参数
# 2. batched=True：函数收到一批（examples 是 list），更快
# 3. map 的 batch（默认1000）≠ 训练的 batch（如8），两码事
# 4. 为什么用 map：Arrow 格式存磁盘，按需加载，省内存
# 5. 为什么 map 时不 padding：留给 DataCollator 动态做，更省
# 6. 为什么必须 rename：模型/Trainer 认 "labels" 这个名字
# 7. set_format("torch")：list → 张量，才能喂模型
#
# ============================================================