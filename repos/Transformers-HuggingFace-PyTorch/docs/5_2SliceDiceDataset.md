# Time to slice and dice

## Slice and dice a dataset 

- Most of the time the data you work with, won't be perfectly prepared for training models.
- We will explore the various features that the datasets library provides to clean up your data.
- The Datasets library provides several methods to filter and transform a dataset.
    - Shuffle and split
    - Select and filter
    - Rename, remove and flatten
    - Map

### Shuffle and split

- You can easily shuffle the whole dataset with Dataset.shuffle()

``` py
from datasets import load_dataset

squad = load_dataset("squad", split="train")
squad[0]

squad_shuffled = squad.shuffle(seed=666)
squad_shuffled[0]
```

- It is genrally a good idea to place shuffling to your training set so that your model doesn't learn any artificial ordering the data.

- Another way to shuffle the data is to create random train and test splits.
- This can be useful if you have to create your own test splits from raw data.
- To do this, you just apply the train_test_split() and specify how large the test split should be.

``` py
dataset = squad.train_test_split(test_size=0.1)
dataset
```

### Select and filter

- You can return rows according to a list of indices using Dataset.select().
- This method expects a list or a generator of the dataset's indices and will then return a new dataset object containing just those rows.

``` py
indices = [0, 10, 20, 40, 80]
examples = squad.select(indices)
examples
```

- If you want to create a random sample of rows, you can do this by chaining the shuffle and select methods together.

``` py
sample = squad.shuffle().select(range(5))
sample
```

- The last way to pick out specific rows in a dataset is by applying the filter().
- This method checks whether each row fulfills some condition or not.

``` py
squad_filtered = squad.filter(lambda x : x["title"].startswith("L"))
squad_filtered[0]
```

### Rename, remove and flatten

- Use the rename_column() and remove_column() methods to transform your columns.
- rename_column() to change the name of the column.

``` py
squad.rename_column("context", "passages")
squad
```

- remove_column() to delete them

``` py
squad.remove_columns(["id", "title"])
squad
```

- Some datasets have nested columns and you can expand these by applying the flatten()

``` py
squad.flatten()
squad
```

### Map method

- the Dataset.map() method applies a custom processing function to each row in the dataset

``` py
def lowercase_title(example):
    return {"title": example["title"].lower()}

squad_lowercase = squad.map(lowercase_title)
# Peek at random sample
squad_lowercase.shuffle(seed=42)["title"][:5]
```

- The map() method can also be used to feed batches of rows to the processing function.
- This is especially useful for tokenization where the tokenizers are backed by the tokenizer library and they can use fast multithreading to process batches in parallel. 

``` py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_title(example):
    return tokenizer(example["title"])

squad.map(tokenize_title, batched=True, batch_size=500)
```

## From Datasets to DataFrames and back

- Although the processing functions of the datasets library will cover most of the cases needed to train a model, there are times when you will need to switch to a library like pandas to access more powerful features or high-level apis for visualizations.
- Fortunately the datasets library is designed to be inter-operable with libraries like pandas, numpy, pytorch, tensorflow and jax.
- By default, a Dataset object will return Python objects when you index it.

``` py
from datasets import load_dataset
dataset = load_dataset("swiss_judgment_prediction", "all_languages", split="train")
dataset[0]
```

- but what can you do if you want to ask complex questions about the data?
- Suppose that before we train any models we would like to explore the data a bit.
- Explore questions like which legal area is the most common? or How are the languages distributed across regions?
- Luckily we can use Pandas to answer these questions! Since answering these questions with the native arrow format is not easy.
- The way this works is that by using the set_format(), we will change the output format of the dataset from python dictionaries to pandas dataframes.

``` py
# Convert the output format to pandas.DataFrame
dataset.set_format("pandas")
dataset[0]
```

- the way this words under the hood is that the datasets library changes the magic method __getitem__() of the dataset.
- the __getitem__() method is a special method for python containers that allows you to specify how indexing works.
- In this case, the __getitem__() method of the raw dataset starts off by returning a python dictionary and then after applying the set_format() method we change __getitem__() method to return dataframes instead.

``` py
dataset.__getitem__(0)
dataset.set_format("pandas")
dataset.__getitem__(0)
```

- Another way to create a Dataframe is with the Dataset.to_pandas() method.
- The datasets library also provides a to_pandas() method if you want to do the format conversion and slicing of the datset in one go.

``` py
df = dataset.to_pandas()
df.head()
```

- Once we have a DataFrame, we can query our data, make pretty plots and so on.

``` py
# How are languages distributed across regions?
df.groupby("region")["language"].value_counts()

# Which legal area is most common?
df["legal area"].value_counts()
```

- Just remember to reset the format back to arrow tables when you are finished.
- If you don't you can run into the problems if you try to tokenize your text, because it is no longer represented as strings in a dictionary

``` py
from transformers import AutoTokenizer

# Load a pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Tokenize the `text` column
dataset.map(lambda x : tokenizer(x["text"]))
```

- By resetting the output format, we get back arrow tables and we can tokenize without problems.

``` py
# Reset back to Arrow format
dataset.reset_format()
# Now we can tokenize!
dataset.map(lambda x : tokenizer(x["text"]))
```

## Saving and reloading a dataset

- When you download or process or process a dataset, the processing scripts and data are stored locally in a cache directory.
- The cache allows the datasets library to avoid re-downloading or processing the entire dataset everytime you use it.
- Now the data is stored in the form of arrow tables whose location can be found by accessing the dataset's cache_files attribute.

``` py
from datasets import load_dataset
raw_datasets = load_dataset("allocine")
raw_datasets.cache_files
```

- Alternatively, you can save your dataset in a different location and format.
- The Datasets library provides 4 main functions to achieve this:
    - Arrow: Dataset.save_to_disk()
    - CSV: Dataset.to_csv()
    - JSON: Dataset.to_json()
    - Parquet: Dataset.to_parquet()
- CSV and JSON formats are great if you just want to quickly save a small or medium sized dataset.
- But if your dataset is huge, you will want to save it in either the arrow or parquet formats.
- Arrow files are great if you plan to reload or process the data in the near future.
- While parquet files are designed for long term storage and are very space efficient.

### Arrow format

- When you save in Arrow format, each split and its metadata are stored in a separate directory.
- Here we provide the path we wish to save the data to and the datasets library will automatically create a directory for each split to store the arrow table and the metadata.
- Since we are dealing with a dataset dict object that has multiple splits, this information is also stored in the dataset_dict.json file.

``` py
raw_datasets.save_to_disk("my-arrow-datasets")
my-arrow-datasets/
├── dataset_dict.json
├── test
│   ├── dataset.arrow
│   ├── dataset_info.json
│   └── state.json
├── train
│   ├── dataset.arrow
│   ├── dataset_info.json
│   ├── indices.arrow
│   └── state.json
└── validation
    ├── dataset.arrow
    ├── dataset_info.json
    ├── indices.arrow
    └── state.json
```

- To reload a local Arrow dataset, we use the load_from_disk() function.
- We simply pass the path of our dataset directory and the original dataset will be recovered.

``` py
from datasets import load_from_disk
arrow_datasets_reloaded = load_from_disk("my-arrow-datasets")
arrow_datasets_reloaded
```

### CSV Format

- To save a dataset in CSV format, we save each split as a separate file.
- We use to_csv() function.
- In this case, you will need to loop over the splits of the dataset dict object and save each dataset as an individual csv file.
- Since the o_csv() function is based on the one from pandas, you can pass keyword arguments to configure the output 

``` py
for split, dataset in raw_datasets.items():
    dataset.to_csv(f"my-dataset-{split}.csv", index=None)
```

- We can then reload the CSV files using the data_files argument, together with the csv loading script and the data files argument which specifies the file names associated with each split 

``` py
data_files = {
    "train": "my-dataset-train.csv",
    "validation": "my-dataset-validation.csv",
    "test": "my-dataset-test.csv",
}
csv_datasets_reloaded = load_dataset("csv", data_files=data_files)
csv_datasets_reloaded
```

### JSON & Parquet Formats

- Saving a dataset in JSON and Parquet formats is similar to CSV

``` py
# Save in JSON Lines format
for split, dataset in raw_datasets.items():
    dataset.to_json(f"my-dataset-{split}.jsonl")

# Save in Parquet format
for split, dataset in raw_datasets.items():
    dataset.to_parquet(f"my-dataset-{split}.parquet")
```

- We also reload JSON and Parquet files using the data_files argument.

``` py
json_data_files = {
    "train": "my-dataset-train.jsonl",
    "validation": "my-dataset-validation.jsonl",
    "test": "my-dataset-test.jsonl",
}
parquet_data_files = {
    "train": "my-dataset-train.parquet",
    "validation": "my-dataset-validation.parquet",
    "test": "my-dataset-test.parquet",
}
# Reload with the `json` script
json_datasets_reloaded = load_dataset("json", data_files=json_data_files)
# Reload with the `parquet` script
parquet_datasets_reloaded = load_dataset("parquet", data_files=parquet_data_files)
```
