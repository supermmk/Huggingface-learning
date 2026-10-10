from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("code_search_net", "python")

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples =  dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]

training_corpus = get_training_corpus()

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

tokenizer.save_pretrained("code-search-net-tokenizer")