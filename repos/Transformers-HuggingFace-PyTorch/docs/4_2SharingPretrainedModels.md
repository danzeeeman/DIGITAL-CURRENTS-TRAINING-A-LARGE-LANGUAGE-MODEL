# Sharing pretrained models
- Below are the easiest ways to share pretrained models to the HuggingFace Hub.
- There are three ways to go about creating new model repositories:
    - Using the push_to_hub API
    - Using the huggingface_hub Python library
    - Using the web interface

- Once you’ve created a repository, you can upload files to it via git and git-lfs.

## Managing a repo on the Model Hub

- In order to handle a repository, you should first have a huggingface account: https://huggingface.co/join
- Once you are logged in, you can create a new repository by clicking on the new model option: https://huggingface.co/new
    - Owner: pallavi176  # your namespace or your organization namespace
    - Model name: dummy_model3
    - License: mit
    - public (recommended free option)
    - Click on create model
    - Files and versions tab is kind of git repo (version contol)

### Adding files to the repository

- In Files and versions tab, files can be added through the web interface through add file button.
    - Click Add file -> then Create a new file
    - files: next_file.txt (can be of format txt, json, etc)
    - content: new file (can add content to the file)
 - Can add files using huggingface_hub library and through command line

#### Upload files using huggingface_hub library
 - login to your account: https://huggingface.co/pallavi176

``` py
!pip install huggingface_hub
```

``` py
from huggingface_hub import notebook_login
# Login with huggingface write access token
notebook_login()
```

- Upload file using upload_file() method:

``` py
from huggingface_hub import upload_file
# upload_file("Current loaction of the file", 'path of the file in repo', 'id of the repo we are pushing')
upload_file("path_to_file", 'path_in__file in _repo', '<namespace>/<repo_id>')
```
- Additional parameters:
    - token: if you would like to specify a different token than the one saved in your cache with your login
    - repo_type: if you would loke to push to a 'dataset' or a 'space'

- Upload Readme file:
``` py
with open("/tmp/README.md", "w+") as f:
  f.write("# My dummy model")
upload_file(path_or_fileobj="/tmp/README.md", path_in_repo="README.md", repo_id="pallavi176/dummy_model3")
```
##### delete_file() method to delete the file from repo

``` py
from huggingface_hub import delete_file
delete_file(path_in_repo="README.md", repo_id="pallavi176/dummy-model2")
```

- This approach using only these 2 methods is super simple.
- It doesn't need git or git lfs installed
- Limitation: The maximum file size that can be uploaded is limited to GB

#### Uploading using repository utility

- This class is a wrapper over git and git lfs methods which abstracts most of the complexity and offers a flexible api to manage your online repositories

``` py
from huggingface_hub import Repository
repo = Repository("local-folder", clone_from="pallavi176/dummy_model3")
```

- Cloned from "pallavi176/dummy_model3"  huggingface repository to local directory: "local-folder"
- Upload a trained model from local:

``` py
from transformer import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("/tmp/cool-model")
tokenizer = AutoTokenizer.from_pretrained("/tmp/cool-model")
repo.git_pull()
```

- Save the model & tokenizer files inside that folder

``` py
model.save_pretrained(repo.local_dir)
tokenizer.save_pretrained(repo.local_dir)
```

- We will start the file by adding git add method:

``` py
repo.git_add()
repo.git_commit("Added model and tokenizer")
repo.git_push()
repo.git_tag()
```

## The Push to Hub API (PyTorch)

- Login to huggingface using token id:

``` py
from huggingface_hub import notebook_login
# Login with huggingface write access token
notebook_login()
```

- Launch fine tuning of bert model using gule cola dataset:

``` py
from datasets import load_dataset, load_metric
raw_datasets = load_dataset("glue", "cola")
from transformers import AutoTokenizer
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
import numpy as np
from datasets import load_metric
metric = load_metric("glue", "cola")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

- Push to hub by setting parameter in the training argument: push_to_hub=True

``` py
from transformers import TrainingArguments
args = TrainingArguments(
    "bert-fine-tuned-cola",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)
```

- This will automatically upload your model to the app each time it is saved, so every epoch in our case
- We can choose which model_id to push to using argument hub_model_id="other name"

``` py
from transformers import TrainingArguments
args = TrainingArguments(
    "bert-fine-tuned-cola",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
    hub_model_id="other name"
)
```

- We can launch training and it will upload at every epoch as mentioned in the training argument

``` py
from transformers import Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()
)
trainer.push_to_hub("End of training")
```

### Pusing components individually

- If you are not using Trainer API to train your model, you can use push_to_hub() on model & tokenizer directly

``` py
repo_name = "bert-fine-tuned-cola"
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
```

### To fix the existing labels on the hub:

``` py
label_names = raw_datasets["train"].features["label"].names
model.config.id2label = {str(i): lbl for i, lbl in enumerate(label_names)}
model.config.label2id = {lbl: str(i) for i, lbl in enumerate(label_names)}
repo_name = "bert-fine-tuned-cola"
model.config.push_to_hub(repo_name)
```

### Use uploaded model

``` py
from transformers import pipeline
classifier = pipeline("text-classification", model="pallavi176/bert-fine-tuned-cola")
classifier("This is incorrect sentence.")
```

