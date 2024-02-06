- We’ve explored 
    - how tokenizers work and looked at tokenization, 
    - conversion to input IDs, 
    - padding, 
    - truncation, 
    - attention masks.

## Tokenization

- When we call your tokenizer directly on the sentence, you get back inputs that are ready to pass through your model:

``` py
from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
```
- Here, the model_inputs variable contains everything that’s necessary for a model to operate well.

- tokenize a single sequence:
``` py
# tokenize a single sequence
sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
```

- tokenize multiple sequences at a time:
``` py
# tokenize multiple sequences at a time
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
model_inputs = tokenizer(sequences)
```

- It can pad according to several objectives:
``` py
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")
# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")
# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

- It can also truncate sequences:
``` py
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)
# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

- The tokenizer object can handle the conversion to specific framework tensors, which can then be directly sent to the model. 
- Prompting the tokenizer to return tensors from the different frameworks — "pt" returns PyTorch tensors, "tf" returns TensorFlow tensors, and "np" returns NumPy arrays:

``` py
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")
# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

### Special tokens

``` py
sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
```

``` py
# decode the tokens
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))
# "[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
# "i've been waiting for a huggingface course my whole life."
```

- The tokenizer added the special word [CLS] at the beginning and the special word [SEP] at the end.
- This is because the model was pretrained with those, so to get the same results for inference we need to add them as well.
- Note that some models don’t add special words, or add different ones; models may also add these special words only at the beginning, or only at the end.

## Wrapping up: From tokenizer to model

- one final time how it can handle multiple sequences (padding!), very long sequences (truncation!), and multiple types of tensors with its main API:

``` py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```
