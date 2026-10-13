# ============================================================
# BPE 算法完整实现（教学版）
# 目标：从语料学出合并规则，再用规则分词
# ============================================================

from collections import defaultdict
from transformers import AutoTokenizer

# ============================================================
# 一、准备语料 + 预分词器
# ============================================================
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ============================================================
# 二、统计词频（训练第 1 步）
# ============================================================
word_freqs = defaultdict(int)

for text in corpus:
    # 用 GPT-2 的预分词器切词
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1
# 结果：{'This': 3, 'Ġis': 2, 'Ġthe': 1, ...}


# ============================================================
# 三、构建初始词表（训练第 2 步）
# ============================================================
# 3.1 收集所有出现过的字符
alphabet = []
for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

# 3.2 加特殊 token
vocab = ["<|endoftext|>"] + alphabet.copy()

# 3.3 把每个词拆成字符列表（BPE 起点）
splits = {word: [c for c in word] for word in word_freqs.keys()}


# 结果：{'This': ['T','h','i','s'], 'Ġis': ['Ġ','i','s'], ...}


# ============================================================
# 四、统计相邻对频次（训练第 3 步）
# ============================================================
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq  # ← 加权：乘以词频
    return pair_freqs


# ============================================================
# 五、合并操作（训练第 4 步）
# ============================================================
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]  # 合并
            else:
                i += 1
        splits[word] = split
    return splits


# ============================================================
# 六、主循环：反复合并最高频对（训练第 5 步）
# ============================================================
merges = {}  # 学到的合并规则
vocab_size = 50  # 目标词表大小

while len(vocab) < vocab_size:
    # 6.1 统计相邻对
    pair_freqs = compute_pair_freqs(splits)

    # 6.2 找最高频的 pair
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq

    # 6.3 合并
    splits = merge_pair(*best_pair, splits)

    # 6.4 记录规则 + 加进词表
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])


# 结果：merges 学出 19 条规则，vocab 达到 50


# ============================================================
# 七、用学到的规则分词（推理阶段）
# ============================================================
def tokenize(text):
    # 7.1 预分词
    pre_tokenize_result = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]

    # 7.2 每个词拆成字符
    splits = [[l for l in word] for word in pre_tokenized_text]

    # 7.3 按 merges 规则（顺序）合并
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[idx] = split

    # 7.4 展平输出
    return sum(splits, [])


# ============================================================
# 八、测试
# ============================================================
print(tokenize("This is not a token."))
# ['This', 'Ġis', 'Ġ', 'n', 'o', 't', 'Ġa', 'Ġtoken', '.']