# Building a tokenizer, block by block

## Building a new tokenizer

- To build your tokenizer you need to design all its components:
    - Normalization
    - Pre-tokenization
    - Model
    - Post-Processing
    - Decoding

- In a fast tokenizer all the components are gathered in the backend_tokenizer which is an instance of Tokenizer from the HuggingFace tokenizer library.

``` py
from transformers import AutoTokenizerFast
tokenizer = AutoTokenizerFast.from_pretrained("...")
type(tokenizer.backend_tokenizer)
```

- tokenizers.Tokenizer <= Tokenizer from HuggingFace library.

- The main steps to create your own tokenizer:
    1. Gather a corpus
    2. Create a backend_tokenizer with HuggingFace tokenizers
    3. Load the backend_tokenizer in a HuggingFace transformers tokenizer

- Let's try to rebuild a BERT tokenizer together!

### 1. Gather a corpus

``` py
from datasets import load_dataset
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
```

### 2. Build a BERT tokenizer with HuggingFace tokenizers

``` py
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors, decoders
```

- Initialize a model

``` py
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
```

- Define a normalizer

``` py
tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace(Regex(r"[\p{Other}&&[^\n\t\r]]"), ""),
        normalizers.Replace(Regex(r"[\s]"), " "),
        normalizers.Lowercase(),
        normalizers.NFD(), normalizers.StripAccents()]
)
```

- Define pre-tokenization

``` py
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
```

- Define a trainer

``` py
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
```

- Train with an iterator

``` py
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
```

- Define a template processing class

``` py
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)
```

- Define a decoder

``` py
tokenizer.decoder = decoders.WordPiece(prefix="##")
```

- Load your tokenizer into a HuggingFace transformers tokenizer
