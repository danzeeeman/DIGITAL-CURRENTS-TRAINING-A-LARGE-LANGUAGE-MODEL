# Normalization and pre-tokenization

## What is normalization?

- The normalization step involves some general cleanup, such as removing needless whitespace, lowercasing, and/or removing accents.
- Here is the result of the normalization of several tokenizers on the same sentence.

``` py
FNetTokenizerFast.from_pretrained("google/fnet-base")
RetriBertTokenizerFast.from_pretrained("yjernite/retribert-base-uncased") # and so on
```

- Fast tokenizers provide easy access to the normalization operation.

``` py
from transformers import AutoTokenizer
text = "This is a text with àccënts and CAPITAL LETTERS"
tokenizer = AutoTokenizerFast.from_pretrained('distilbert-base-uncased')
print(tokenizer.backend_tokenizer.normalizer.normalize_str(text))
```

- And it's really handy that the normalization operation is automatically included when you tokenize a text.

``` py
# With saved normalizer
from transformers import AutoTokenizer
text = "This is a text with àccënts and CAPITAL LETTERS"
tokenizer = AutoTokenizer.from_pretrained("albert-large-v2")
print(tokenizer.convert_ids_to_tokens(tokenizer.encode(text)))
```

``` py
# Without saved normalizer
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/albert-tokenizer-without-normalizer")
print(tokenizer.convert_ids_to_tokens(tokenizer.encode(text)))
```

- Some normalizations may not be visible to the eye and but change many things for the computer.
- There are some Unicode normalization standards: NFC, NFD, NFKC and NFKD

- But beware, not all normalizations are suitable for all corpus.

``` py
from transformers import AutoTokenizer
text = "un père indigné"
tokenizer = AutoTokenizerFast.from_pretrained('distilbert-base-uncased')
print(tokenizer.backend_tokenizer.normalizer.normalize_str(text))
```

## What is pre-tokenization?

- The pre-tokenization applies rules to realize a first split of the text.
- Let's look at the result of the pre-tokenization of several tokenizers.: "3.2.1: let's get started!"
    - 'gpt2':               3 . 2 . 1 : Ġlet 's Ġget Ġstarted !
    - 'albert-base-v1':     _3.2.1: _let's _get _started!
    - 'bert-base-uncased':  3 . 2 . 1 : let ' s get started !
- Pre-tokenization can modify text - such as replacing a space with a special underscore - and split text into tokens.

- Fast tokenizers provide easy access to the pre-tokenization operation.

``` py
from transformers import AutoTokenizerFast
tokenizer = AutoTokenizerFast.from_pretrained('albert-base-v1’)
text = "3.2.1: let's get started!"
print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text))
```

