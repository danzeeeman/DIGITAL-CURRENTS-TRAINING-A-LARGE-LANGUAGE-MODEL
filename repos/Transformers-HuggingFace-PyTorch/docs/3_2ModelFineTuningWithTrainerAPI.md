## The Trainer API:

- The transformers provide a Trainer API to easily train or fine-tune Transformer models on your dataset.
- The trainer class secure datasets, your model as well as the training hyperparameters and 
- can perform the training on any kind of setup(cpu, gpu) and
- can also compute the predictions on any datasets and
- if you provide metrics, it will evaluate your model on any dataset
- You can also involve the final data processing such as dynamic padding if you provide tokenizer or given data collator

### Preprocessing:

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
data_collator = DataCollatorWithPadding(tokenizer)
```

- We do not apply padding during preprocessing as we will use dynamic padding with our "DataCollatorWithPadding" method.
- Note that we don't do the final steps of renaming, removing columns or set the format to torch tensors. As the Trainer will do all of these automatically for us by analyzing the model's signature.

### Model

- We also need a model and some training arguments brfore creating the Trainer.
- TrainingArguments class only takes a path to a folder where results and checkpoints will be saved.

``` py
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

``` py
from transformers import TrainingArguments
training_args = TrainingArguments("test-trainer")
```

- You can also customize all the other parameters that your trainer will use- learning rate, num of training epochs, etc.

``` py
from transformers import TrainingArguments
training_args = TrainingArguments(
    "test-trainer",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
)
```

### Training:

- We can then pass everything to the Trainer class and start training.

``` py
from transformers import Trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
```

- The result will display training loss which doesn't really tell anything about how well/worse your model is performing.
- This is because we didn't specify any metric for the evaluation.

### Predictions:

- The predict method allows us to get the predictions of our model on a whole dataset. We can then use those predictions to compute metrics.

``` py
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
# (408, 2) (408,)
```

- It returns a named tuple with 3 fields: 
    - predictions: contains the model predictions
    - label_ids: contains the labels if your dataset have them
    - metrics: it is empty here (we are trying to do that).
- Predictions are the logits of the models for all the sentences in the datset of shape: (408, 2)
- To match them with our labels, we need to take the maximum logits for each predictions to know which of the 2 classes was predicted.
- We do this with argmax()
- Then can use the matrix from dataset libray with load_metric() and it returns the evaluation metric used for the dataset.

``` py
import numpy as np
from datasets import load_metric
metric = load_metric("glue", "mrpc")
preds = np.argmax(predictions.predictions, axis=-1)
metric.compute(predictions=preds, references=predictions.label_ids)
# {'accuracy': 0.8627450980392157, 'f1': 0.9050847457627118}
```

- We can see our model did learn something as it is 86.5% accurate.

- To monitor evaluation metrics during training, we need to define a compute_metrics() (as in above steps) and pass it to the trainer.
- It takes the named tuple with predictions and labels and must return a dictionary of the metrics we want to keep track of.
- By passing the epoch evaluation strategy to our training arguments, we tell the trainer to evaluate at the end of every epoch.

``` py
metric = load_metric("glue", "mrpc")
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

``` py
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
```

