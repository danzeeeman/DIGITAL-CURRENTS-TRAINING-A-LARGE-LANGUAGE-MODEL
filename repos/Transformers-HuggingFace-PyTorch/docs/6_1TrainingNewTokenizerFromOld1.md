# Training a new tokenizer from an old one

- Training a tokenizer is not the same as training a model! Model training uses stochastic gradient descent to make the loss a little bit smaller for each batch. It’s randomized by nature (meaning you have to set some seeds to get the same results when doing the same training twice). 
- Training a tokenizer is a statistical process that tries to identify which subwords are the best to pick for a given corpus, and the exact rules used to pick them depend on the tokenization algorithm. It’s deterministic, meaning you always get the same results when training with the same algorithm on the same corpus.

## Training a new tokenizer

- What is the purpose of training a tokenizer 
- What are the key steps to follow
- What is the easiest way is to it.
- Should I train a new tokenizer when you plan to train a new model from scratch.
- You may want to consider training a new tokenizer so that you have a tokenizer suitable for the training corpus used to train a language model from scratch.
- A tokenizer will not be suitable if it has been trained on a corpus that is not similar to the one you will use to train your model from scratch.
- A trained tokenizer will not be suitable for your corpus if your corpus is in a different language, use new characters such as accent, etc, and a specific vocabulary, for instance medical or legal, or use a different style, eg, language of another country.
- Dissimilarities can arise from:
    - New language
    - New characters
    - New domain
    - New style
- If I take the tokenizer trained on bert-based-uncased model on ignore and normalized step, then we can say that the tokenization operations in the english sentence, "here is a sentence adapted to our tokenizer", produces a set of english tokens.

``` py
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained(
  'huggingface-course/bert-base-uncased-tokenizer-without-normalizer'
)
```

- Here, a sentence of 8 words produces a tokens list of length 9.

``` py
text = "here is a sentence adapted to our tokenizer"
print(tokenizer.tokenize(text))
```

- You may want to consider training a new tokenizer so that you have a tokenizer suitable for the training corpus used to train a language model from scratch.
- If I use the same tokenizer on the sentence in bengali, the words will be divided into may sub-tokens since tokenizes does not know the unicode characters.

``` py
text = "এই বাক্যটি আমাদের টোকেনাইজারের উপযুক্ত নয়"
print(tokenizer.tokenize(text))
```

- The fact that the common word is split into many sub-tokens can be problemmatic.
- Because language models can only handle a sequence of tokens of limited length.
- The tokens that are excessively split the initial text may even impact the performance of the model.
- [UNK] tokens are also problemmatic because the model will not be able to extract any information from them.

- Another example, we see tokenizers replaces words containing characters on `àccënts` or `CAPITAL LETTERS` with [UNK] tokens

``` py
text = "this tokenizer does not know àccënts and CAPITAL LETTERS"
print(tokenizer.tokenize(text))
```

- Finally, if we use this tokenizers to tokenize medical vocabulary, we see that a single word is divided into many smaller tokens.

``` py
text = "the medical vocabulary is divided into many sub-token: paracetamol, phrayngitis"
print(tokenizer.tokenize(text))
```

- Most of the tokenizers used by the current State-of-the-Art language models need to be trained on a corpus that is allowed to one used to pre-train a language model.
- This training consists of learning rules to divide text into tokens 
- On the way to learn these rules the token use them depends on the chosen tokenizer model.

- The procedural for training a tokenizer can be summarized in these main steps:
    1. Gathering a corpus of text
    2. Choosen a tokenizer architecture
    3. Train the tokenizer on the corpus
    4. Save the result 

- Let's say you wanted to train a GPT-2 model on python code.

``` py
example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias
    """

print(old_tokenizer.tokenize(example))
print(new_tokenizer.tokenize(example))
```

- The Transformers library provides a very easy to use method to train a tokenizer using a known architecture on a new corpus.

``` py
AutoTokenizer.train_new_from_iterator(
    text_iterator,
    vocab_size,
    new_special_tokens=None,
    special_tokens_map=None,
    **kwargs
)
```

- The first step is to gather a training corpus.

``` py
from datasets import load_dataset
raw_datasets = load_dataset("code_search_net", "python")
```

``` py
def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]
```

- Then, to train the tokenizer on this new corpus

``` py
training_corpus = get_training_corpus()
```

- We can load GPT2 tokenizer architecture

``` py
from transformers import AutoTokenizer
training_corpus = get_training_corpus()
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

- The 4th line will train on new corpus.

``` py
from transformers import AutoTokenizer
training_corpus = get_training_corpus()
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
```

- Once training is finished, we just have to save the tokenizer locally or send it to the hub.

``` py
from transformers import AutoTokenizer
training_corpus = get_training_corpus()
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
new_tokenizer.save_pretrained("code-search-net-tokenizer")
```

- And we can finally verify that our new tokenizer is more suitable for tokenizing python functions than the original GPT-2 tokenizer.

``` py
example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias
    """

print(old_tokenizer.tokenize(example))
print(new_tokenizer.tokenize(example))
```

