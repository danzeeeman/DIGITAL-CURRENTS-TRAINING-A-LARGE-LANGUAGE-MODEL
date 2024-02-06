# Big data? HuggingFace Datasets to the rescue!

- We will take a look at 2 core features of the Datasets library that allow to load and process huge datasets without blowing up laptop, cpu.
- If you want to train a model from scratch, you will need a LOT of data.
- Nowadays it is not uncommon to find yourself with multi GB-sized datasets, especially if you are planning to pre-train a transformer like Bert or GPT2 from scratch.
- In these cases, even loading the data can be a challenge.
- For example, the C4 corpus used to pretrained T5 consists of over 2 TB of data.
- To handle these large datasets, the Datasets library is built on 2 core features: apache Arrow format and streaming api.
- Datasets library uses Arrow and streaming to handle data at scale.
- Arrow is designed for high performance data processing and represents each table-like dataset with a column-based format.
- eg here: column-based formats group the elements of a table in consecutive blocks on RAM and this unlocks fast access and processing.

### Arrow

- Arrow is great at processing data at any scale but some datasets are so large that you can not even fit them on your hard disk.
- So for these cases, the Datasets library provides a streaming API that allows you to progressively download the raw data one element at a time.
- The result is a special object called an iterable dataset.
- Arrow's memory-mapped format enables access to bigger-than-RAM datasets.
- Arrow is powerful. Its first feature is that it treats every dataset as a memory-map file.
- Now memory-mapping is a mechanism that maps a portion of a file or an entire file and disk to a chunk of virtual memory.
- This allows applications to access segments of an extremely large file without having to read the whole file into memory first.

``` py
from datasets import load_dataset
data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
large_dataset = load_dataset("json", data_files=data_files, split="train")
size_gb = large_dataset.dataset_size / (1024 ** 3)
print(f"Dataset size (cache file) : {size_gb:.2f} GB")
```

``` py
import psutil
# Process.memory_info is expressed in bytes, so convert to megabytes
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
```

- Memory-mapped files can be shared across multiple processes.
- Another cool feature of Arrow is memory-mapping capabilities is that it allows multiple processors to work with the same large dataset without moving it or copying it in any way.
- This zero copy feature of Arrow makes it extremely fast for iterating over a dataset.
- Below we iterate over 15 Million rows in about a minute just using a standard laptop.

``` py
import timeit
code_snippet = """batch_size = 1000
for idx in range(0, len(large_dataset), batch_size):
    _ = large_dataset[idx:idx + batch_size]
"""
time = timeit.timeit(stmt=code_snippet, number=1, globals=globals())
print(
    f"Iterated over {len(large_dataset)} examples (about {size_gb:.1f} GB) in "
    f"{time:.1f}s, i.e. {size_gb/time:.3f} GB/s"
)
```

### Streaming

- Streaming lets you process bigger-than-disk datasets.
- How to stream a large dataset - the only change need to make is to set the streaming=True argument in the load_dataset() function.
- This will return a special iterable dataset object which is a bit different to the dataset objects we have seen till now.
- This object is an iterable which means we can not index it to access elements but instead we iterate on it using iter() and next() methods.
- This will download and access a single example from the dataset which means we can progressively iterate through a huge dataset without having to download it first.

``` py
large_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True)
next(iter(large_dataset_streamed))
```

``` py
type(large_dataset_streamed)
```

- Tokenization with IterableDataset.map() works the same way too.
- Tokenizing text with the map() method also works in a similar way.
- We first stream the dataset and then apply the map() method with the tokenizer.
- To get the first tokenized example, we apply iter() and next() methods.

``` py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = large_dataset_streamed.map(lambda x: tokenizer(x["text"]))
next(iter(tokenized_dataset))
```

- ... but instead of select() we use take() and skip()
- The main difference with an iterable dataset is that instead of using a select() to return examples, we use take() and skip() because we can not index into the dataset.
- The take() method returns the first n examples in the dataset while skip() method skips the first n and retirns the rest.

``` py
# Select the first 5 examples 
dataset_head = large_dataset_streamed.take(5)
list(dataset_head)
```

``` py
# Skip the first 1,000 examples and include the rest in the training set
train_dataset = large_dataset_streamed.skip(1000)
# Take the first 1,000 examples for the validation set
validation_dataset = large_dataset_streamed.take(1000)
```
