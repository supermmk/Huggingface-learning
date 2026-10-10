from transformers import AutoTokenizer
from datasets import load_dataset

# 加载这个可能需要几分钟的时间,你可以趁此喝杯咖啡或茶!
raw_datasets = load_dataset("code_search_net", "python")
# 1. 加载旧 tokenizer（提供算法和配置）
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. 定义语料迭代器（一次产出一批，省内存）
def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]

training_corpus = get_training_corpus()

# 3. 训练新 tokenizer
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

# 4. 保存
tokenizer.save_pretrained("code-search-net-tokenizer")