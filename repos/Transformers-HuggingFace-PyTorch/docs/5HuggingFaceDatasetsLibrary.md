# HuggingFace Datasets Library

- In section 3 you got your first taste of the HuggingFace Datasets library and saw that there were three main steps when it came to fine-tuning a model:
    - Load a dataset from the Hugging Face Hub.
    - Preprocess the data with Dataset.map().
    - Load and compute metrics.

- But this is just scratching the surface of what HuggingFace Datasets can do! In this chapter, we will take a deep dive into the library. Along the way, we’ll find answers to the following questions:
    - What do you do when your dataset is not on the Hub?
    - How can you slice and dice a dataset? (And what if you really need to use Pandas?)
    - What do you do when your dataset is huge and will melt your laptop’s RAM?
    - What the heck are “memory mapping” and Apache Arrow?
    - How can you create your own dataset and push it to the Hub?

- Learned to:
    - Load datasets from anywhere, be it the Hugging Face Hub, your laptop, or a remote server at your company.
    - Wrangle your data using a mix of the Dataset.map() and Dataset.filter() functions.
    - Quickly switch between data formats like Pandas and NumPy using Dataset.set_format().
    - Create your very own dataset and push it to the Hugging Face Hub.
    - Embed your documents using a Transformer model and build a semantic search engine using FAISS.
