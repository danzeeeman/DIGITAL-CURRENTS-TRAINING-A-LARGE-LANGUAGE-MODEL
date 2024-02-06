# What if my dataset isn't on the Hub?

## Loading a custom dataset
- We will explore how the datasets library can be used to load datasets that are not available on the huggingface hub.
- HuggingFace Datasets provides loading scripts to handle the loading of local and remote datasets. It supports several common data formats, such as:
    - load_dataset("csv", data_files="my_file.csv")
    - load_dataset("text", data_files="my_file.txt")
    - load_dataset("json", data_files="my_file.jsonl")
    - load_dataset("pandas", data_files="my_dataframe.pkl")
- To load a dataset in one of above formats, you just need to provide the name of the format to the load_dataset() function alongwith the data_files argument that points to 1 or more file paths or urls.

### Loading from local

- Here's how we can load a local csv dataset.

``` py
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

from datasets import load_dataset
local_csv_dataset = load_dataset("csv", data_files="winequality-white.csv", sep=";")
local_csv_dataset["train"]
```

- Here dataset is loaded automatically as a DatasetDict object with each column in the csv file represented as a feature.

### Loading from remote

- Remote datasets be loaded by passing URLs to the data_files argument

#### Loading csv files

``` py
# Load the dataset from the URL directly
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
remote_csv_dataset = load_dataset("csv", data_files=dataset_url, sep=";") #sep like we pass in pandas dataframe
remote_csv_dataset
```

- Here, the data_files argument points to a url inside of a local file path

#### Loading raw text files

- Raw text files are read line by line to build the dataset

``` py
dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text_dataset = load_dataset("text", data_files=dataset_url)
text_dataset["train"][:5]
```

#### Loading json files

- Json files can loaded in 2 main ways-
    - either line by line: called json lines where every row in the file is a separate json object
        - for these files you can load the dataset by selecting the json loading script and pointing the data_files argument to the file/url

``` py
dataset_url = "https://raw.githubusercontent.com/hirupert/sede/main/data/sede/train.jsonl"
json_lines_dataset = load_dataset("json", data_files=dataset_url)
json_lines_dataset["train"][:2]
```

    - the other json format is by specifying a field in nested JSON
        - these files basically look like one huge dictionary, so the load_dataset() allows you to specify which specific key to load

``` py
dataset_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
json_dataset = load_dataset("json", data_files=dataset_url, field="data")
json_dataset
```

- You can also specify which splits to return with the data_files argument
- If you have more than one split, you can load them by treating data files as a dictionary that maps each split name to its corresponding file.
- Everythingelse stays completely unchanged

``` py
url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
data_files = {"train": f"{url}train-v2.0.json", "validation": f"{url}dev-v2.0.json"}
json_dataset = load_dataset("json", data_files=data_files, field="data")
json_dataset
```

