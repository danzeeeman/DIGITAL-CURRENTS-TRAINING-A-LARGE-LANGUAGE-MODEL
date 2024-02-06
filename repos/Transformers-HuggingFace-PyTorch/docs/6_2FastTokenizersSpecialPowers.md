# Fast tokenizers' special powers

## Why are fast tokenizers called fast?

- We will exactly how much faster the so-called fast tokenizers are compared to the slow tokenizers.
- Let's see how fast tokenizers are!
- Mnli dataset contains 432000 spares of text.

``` py
from datasets import load_dataset
raw_datasets = load_dataset("glue", "mnli")
raw_datasets
```

- We will see how long it takes for the fast and slow versions of a bert tokenizer to process them all.
- We define two functions to preprocess the datasets.
- We define fast and slow tokenizer using AutoTokenizer api.

``` py
from transformers import AutoTokenizer
fast_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_with_fast(examples):
    return fast_tokenizer(
        examples["premise"], examples["hypothesis"], truncation=True
    )
```

- The fast tokenizer is the default when available.
- So we pass along use_fast=False to define the slow one.

``` py
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)
def tokenize_with_slow(examples):
    return fast_tokenizer(
        examples["premise"], examples["hypothesis"], truncation=True
    )
```

- Let's see an example of this on masked language modeling.
- In a notebook, we can time the execution of the cell with %time magic command.
- Processing the whole dataset is 4 times faster with fast tokenizer.
- That's better but very impressive.
- This is because we pass on text to the tokenizer one at a time.
- This is a common mistake to do with fast tokenizers which are backed by Rust.

``` py
%time tokenized_datasets = raw_datasets.map(tokenize_with_fast)
```

``` py
%time tokenized_datasets = raw_datasets.map(tokenize_with_slow)
```

- Properly using a fast tokenizer requires giving it multiple texts at the same time.
- Using fast tokenizers with batched=True is much, much faster.

``` py
%time tokenized_datasets = raw_datasets.map(tokenize_with_fast, batched=True)
```

``` py
%time tokenized_datasets = raw_datasets.map(tokenize_with_slow, batched=True)
```

## Fast tokenizer superpowers

- When performing tokenization, we lose some information.
- eg: here the tokenization is the same for below 2 sentences even if 1 has several more spaces than others.

``` py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(tokenizer("Let's talk about tokenizers superpowers.")["input_ids"])
print(tokenizer("Let's talk about tokenizers      superpowers.")["input_ids"])
```

- It is also difficult to know which word a token belongs to.
- It is difficult to know when 2 or more tokens belong to same word or not.
- Fast tokenizers keep track of the word each token comes from.

``` py
encoding = tokenizer("Let's talk about tokenizers superpowers.")
print(encoding.tokens())
print(encoding.word_ids())
```

- They even keep track of each character span in the original text that gave each token.

``` py
encoding = tokenizer(
    "Let's talk about tokenizers     superpowers.",
    return_offsets_mapping=True
)
print(encoding.tokens())
print(encoding["offset_mapping"])
```

- The internal pipeline of the tokenizer looks like this.
    - Normalization:    "Let's talk about tokenizers     superpowers."
    - Pre-tokenization: [Let,',s,talk,about,tokenizers,superpowers,.]
    - Applying Model:   [Let,',s,talk,about,token,##izer,##s,super,##power,##s,.]
    - Special tokens:   [[CLS],Let,',s,talk,about,token,##izer,##s,super,##power,##s,.,[SEP]]
- The fast tokenizers keep track of the original span of text creating each word or token.
- Here are a few applications of these features:
    - Word IDs application: Whole word masking, Token classification
    - Offset mapping application: Token classification, Question Answering

## Inside the Token classification pipeline (PyTorch)

- The token classification pipeline gives each token in the sentence a label.

``` py
from transformers import pipeline
token_classifier = pipeline("token-classification")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

- It can also group together tokens corresponding to the same entity.

``` py
token_classifier = pipeline("token-classification", aggregation_strategy="simple")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

- The token classification pipeline follows the general steps of the pipeline we saw before.
    - Tokenizer:      Raw text -> Input IDs
    - Model:          Input IDs -> Logits
    - Postprocessing: Logits -> Predictions

- We have already seen the first tseps of the pipeline: tokenization and model.

``` py
from transformers import AutoTokenizer, AutoModelForTokenClassification
model_checkpoint = ""
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)
print(inputs["input_ids"].shape)  # [1,19]
print(outputs.logits.shape)       # [1,19,9]
```

- The model outputs logits, which we need to convert to probabilities using softmax.
- We also get the predicted level for each token by taking the maximum prediction

``` py
import torch
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = probabilities.argmax(dim=-1)[0].tolist()
print(predictions)
```

- The label correspondence then lets us match each prediction to a label.

``` py
model.config.id2label
```

- The start and end character positions can be found using the offset mappings.

``` py
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]
for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {"entity": label, "score": probabilities[idx][pred],
             "word": tokens[idx], "start": start, "end": end}
        )
print(results)
```

- The last step is to group all the tokens corresponding to the same entity together.
- We have to group together in one entity all the corresponding labels.
- We group together tokens with the same label unless it's a B-XXX.

``` py
import numpy as np
label_map = model.config.id2label
results = []
idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = label_map[pred]
    if label != "O":
        # Remove the B- or I-
        label = label[2:]
        start, _ = offsets[idx]
        # Grab all the tokens labeled with I-label
        all_scores = []
        while idx < len(predictions) and label_map[predictions[idx]] == f"I-{label}":
            all_scores.append(probabilities[idx][pred])
            _, end = offsets[idx]
            idx += 1
        # The score is the mean of all the scores of the token in that grouped entity.
        score = np.mean(all_scores).item()
        word = example[start:end]
        results.append(
            {"entity_group": label, "score": score,
             "word": word, "start": start, "end": end}
        )
    idx += 1
```
