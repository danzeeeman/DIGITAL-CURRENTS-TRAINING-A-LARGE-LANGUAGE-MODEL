# Handling multiple sequences

## Batching inputs together:

- Sentences we want to group inside a batch will often have different lengths

``` py
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sentences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this.",
]
tokens = [tokenizer.tokenize(sentence) for sentence in sentences]
ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
print(ids[0])
print(ids[1])
#[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]
#[1045, 5223, 2023, 1012]
```

- You can't build a tensor with lists of different lengths 
- because all arrays and tensors should be rectangular

``` py
import torch
ids = [[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012],
       [1045, 5223, 2023, 1012]]
input_ids = torch.tensor(ids) # ValueError: expected sequence of length 14 at dim 1 (got 4)
```

- Generally, we only truncate sentences when they are longer than the maximum length the model can handle
- Which is why we usually pad the smaller sentences to the length of the longest one!

``` py
import torch
ids = [[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012],
       [1045, 5223, 2023, 1012,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0]]
input_ids = torch.tensor(ids)
input_ids
```

``` py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token_id      # Applying padding here
```

- Now that we have padded our sentences we can make a batch with them
- But just passing this through a transformers model will not give the right results.

``` py
from transformers import AutoModelForSequenceClassification
ids1 = torch.tensor(
    [[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]]
)
ids2 = torch.tensor([[1045, 5223, 2023, 1012]])
all_ids = torch.tensor(
    [[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012],
     [1045, 5223, 2023, 1012,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0]]
)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
print(model(ids1).logits)
print(model(ids2).logits)
print(model(all_ids).logits)
"""
tensor([[-2.7276,  2.8789]], grad_fn=<AddmmBackward>)
tensor([[ 3.9497, -3.1357]], grad_fn=<AddmmBackward>)
tensor([[-2.7276,  2.8789],
        [ 1.5444, -1.3998]], grad_fn=<AddmmBackward>)
"""
```

- This is because the attention layers use the padding tokens in the context they look at for each token in the sentence.
    - Attention layers attend just the 4 tokens: [I, hate, this, !]
    - Attention layers attend the 4 tokens and all padding tokens: [I, hate, this, !, [PAD], [PAD], [PAD], [PAD]]

- To tell the attention layers to ignore the padding tokens, we need to pass them an attention mask.

``` py
all_ids = torch.tensor(
    [[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012],
     [1045, 5223, 2023, 1012,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0]]
)
# adding attention by creating attention mask
attention_mask = torch.tensor(
    [[   1,    1,    1,    1,    1,    1,    1,     1,     1,    1,    1,    1,    1,    1],
     [   1,    1,    1,    1,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0]]
)
```

- Here, attention layers will ignore the tokens marked with 0 in the attention mask.

- With the proper attention mask, predictions are the same for a given sentence, with or without padding.

``` py
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
output1 = model(ids1)
output2 = model(ids2)
print(output1.logits)
print(output2.logits)
output = model(all_ids, attention_mask=attention_mask)
print(output.logits)
```

- Using with padding=True, the tokenizer can directly prepare the inputs with padding and the proper attention mask:

``` py
from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sentences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this.",
]
print(tokenizer(sentences, padding=True))
# {'input_ids': [[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102], [101, 1045, 5223, 2023, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
# 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}
```
