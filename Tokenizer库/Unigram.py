corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

from collections import defaultdict

word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

word_freqs

char_freqs = defaultdict(int)
subwords_freqs = defaultdict(int)
for word, freq in word_freqs.items():
    for i in range(len(word)):
        char_freqs[word[i]] += freq
        # 循环遍历长度至少为2的子字
        for j in range(i + 2, len(word) + 1):
            subwords_freqs[word[i:j]] += freq

# 按频率对子词排序
sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
sorted_subwords[:10]

token_freqs = list(char_freqs.items()) + sorted_subwords[: 300 - len(char_freqs)]
token_freqs = {token: freq for token, freq in token_freqs}

from math import log

total_sum = sum([freq for token, freq in token_freqs.items()])
model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

def encode_word(word, model):
    n = len(word)
    scores = [0] * n
    best_segmentations = [[] for _ in range(n)]
    scores[0] = model[word[0]]
    best_segmentations[0].append(word[0])
    for i in range(1, n):
        whole = word[: i + 1]
        if whole in model
            scores[i] = model[word[:i + 1]]
            best_segmentations[i].append(word[:i + 1])

        for j in range(0 , i):
            score = scores[j] + model[word[j + 1 : i + 1]]
            if score > scores[i] :
                scores[i] = score
                best_segmentations[i] = best_segmentations[j][:]
                best_segmentations[i].append(word[j + 1 : i + 1])


