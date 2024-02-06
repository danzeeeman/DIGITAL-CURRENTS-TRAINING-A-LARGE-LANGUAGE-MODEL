## Write your training loop in PyTorch

### Preprocessing

- Here is how we can easily preprocess the GLUE MRPC dataset using dynamic padding.

``` py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer)
```

- Once our data is preprocessed, we just have to create our DataLoaders which will be responsible to convert our dataset into batches.

``` py
from torch.utils.data import DataLoader
train_dataloader = DataLoader(
  tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
  tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```

- To check everything works as intended, we try to grab a batch of data and inspect it.

``` py
for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})
# {'attention_mask': torch.Size([8, 63]), 'input_ids': torch.Size([8, 63]), 'labels': torch.Size([8]), 'token_type_ids': torch.Size([8, 63])}
```

- Like dataset element it is dictionary, but this time these values are not a single list of integers but a tons of batches of shape batch size by sequence length.

### Model

- The next step is to create our model and send our training data into model.
- We will use from_pretrained() method and adjust the number of labels to the number of classes we have in our dataset(here, 2):

``` py
from transformers import AutoModelForSequenceClassification
checkpoint = "bert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

- To be sure everything is going well, we pass the batch required to our model and check there is no error

``` py
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
# tensor(0.7512, grad_fn=<NllLossBackward>) torch.Size([8, 2])
```

### Training:

- These labels are provided, so the models of the transformers library always returns the loss directly.
- Will use loss.backward() to compute the gradients.
- Then we will use the optimizer to do the training step.
- The optimizer will be responsible for doing the training updates to the model weights.

``` py
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)
```

``` py
loss = outputs.loss
loss.backward()
optimizer.step()
# Don't forget to zero your gradients once your optimizer step is done!
optimizer.zero_grad()
```

- We will add 2 more things to make it as good as it can be.
- The first one is lr schedular. 
- The learning rate schedular will update the optimizer's learning rate at each step.

``` py
from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
```

- You can use any Pytorch leaning rate schedular in place of it.
- We can make training faster by using GPU instead of CPU.

``` py
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)
```

- Putting everything together, here is what the training loop looks like.

``` py
optimizer = AdamW(model.parameters(), lr=5e-5)
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))
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
```

### Evaluation

- Evaluation can then be done like this with a datasets metric.
- First we put our model in the evaluation mode to deactivate layers like dropout, then go through all the evaluation that are needed.
- Model provides logits and we need to provide argmax function to convert them into predictions

``` py
from datasets import load_metric
metric= load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)   
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
metric.compute()
```

-  metric.add_batch() to send to the predictions.

