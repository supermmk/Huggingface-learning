import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler, \
    trainer_optimizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import evaluate

raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
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

tokenized_datasets = tokenized_datasets.remove_columns("sentence1", "sentence2", "idx")
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8,
    collate_fn=data_collator,
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_train_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_scheduler(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

progress_bar = tqdm(train_dataloader)

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()

        progress_bar.update(1)


metric = evaluate.load("nyu-mll/glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())