# Creating your own dataset

## Uploading the dataset to the Hugging Face Hub

### Step 1: Create a repository to host all your files

- The first thing you need to do is create a new dataset repository on the hub.
- So just click on your profile icon and select the New Dataset option.
- Next we need to assign an owner of the dataset. By default, this will be your hub account but you can also create datasets under any organization that you belong to
- Then we just need to give the dataset a good name
- specify whether it is a public or private dataset
- Public datasets can be accessed by anyone while private datasets can only be accessed by you or members of your organization
- With that we can go ahead and create the dataset.

### Step 2: Upload your files

- Now that you have an empty dataset repository on the hub, the next thing to do is add some actual data to it.
- You can do this with git but the easiest way is by selecting the upload file button 
- Then you can just go ahead and upload the files directly from your machine.
- After you have uploaded your files, you will see them appear in the repository under the files and the versions tab.

### Step 3: Create a dataset card

- The last step is to create a dataset card
- Well documented datasets are more likely to be usefuls to others as they provide the context to decide whether the dataset is relevant or whether there are any biases or risks associated with using the dataset.
- On the huggingface hub this information is stored in each repository's Readme file.
- There are 2 main steps that you should take:
- First you neeed to create some metadata that will allow your dataset to be easily found by others on the hub
- You can create this metadata using the dataset's tag and application
- Once you have created the metadata you can fill out the rest of the dataset card 

### Step 4: Load your dataset and have fun!

- Once your dataset is on the hub you can load it using the load_dataset() function
- Just provide the name of your repostory and a data_files and you are good to go

``` py
from datasets import load_dataset
data_files = {"train":"train.csv", "test":"test.csv"}
my_dataset = load_dataset("pallavi176/my-awesome-dataset", data_files=data_files)
my_dataset
```

