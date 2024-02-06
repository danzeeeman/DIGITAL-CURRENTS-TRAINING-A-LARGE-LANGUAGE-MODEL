# Semantic search with FAISS

## Text embeddings & semantic search

- How transformer models represent text as embedding vectors and how these vectors can be used to find similar documents in a corpus.
- Text embeddings represent text as vectors.
- Text Embeddings are just a fancy way of saying that we can represent text as an array of numbers called a vector.
- To create these embeddings, we usually use an encoder-based model like BERT.

- We can use metrics like cosine similarity to compare how close two embeddings are.
- The trick to do the comparison is to compute a similarity metric between each pair of embedding vectors.
- These vectors usually live in a very high dimensional space.
- So a similarity metrics can be anything that measures some sort of distance between vectors.
- One very popular metric is cosine similarity which uses the angle between two vectors to measure how close they are.

- Each token is represented by one vector, but we want one vector per sentence.
- ie, One problem we have to deal with is that transformer models like bert will actually return one embedding vector per token.
- In below example, the output of model has produced 9 embedding vectors per sentence and each vector has 384 dimensions.
- But what we really want is a single embedding vector for each sentence.

``` py
import torch
from transformers import AutoTokenizer, AutoModel
sentences = [
    "I took my dog for a walk",
    "Today is going to rain",
    "I took my cat for a walk",
]
model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    model_output = model(**encoded_input)  
token_embeddings = model_output.last_hidden_state
print(f"Token embeddings shape: {token_embeddings.size()}") # [3,9,384] [num_sentence, num_tokens, embed_dim]
```

- Use mean pooling to create the sentence vectors!
- To deal with this we can use a technique called pooling.
- The simplest pulling method is just to take the token embedding of the special [CLS] token.
- Alternatively we can average the token embeddings which is called mean pooling.
- With mean pooling the only thing we need to make sure is that we don't include the padding tokens in the average which is why you can see the attention mask being used here.
- This gives us a 384 dimensional vector for each sentence which is exactly what we want.

``` py
import torch.nn.functional as F
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
# Normalize the embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
print(f"Sentence embeddings shape: {sentence_embeddings.size()}") # [3,384] [num_sentences, embed_dim]
```

- And once we have our sentence embeddings we can calculate the cosine similarity for each pair of vectors.
- Below, we used sklearn function to calculate cosine similarity.

``` py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
sentence_embeddings = sentence_embeddings.detach().numpy()
scores = np.zeros((sentence_embeddings.shape[0], sentence_embeddings.shape[0]))
for idx in range(sentence_embeddings.shape[0]):
    scores[idx, :] = cosine_similarity([sentence_embeddings[idx]], sentence_embeddings)[0]
```

- You can use the same trick to measure similarity of query against a corpus of docs.
- We can actually take this idea one step further by comparing the similarity between a question and a corpus of documents
- For example, suppose we embed every post in the huggingface forums, we can then ask a question, embed it and check which forum posts are similar.
- This process is often called semantic search because it allows us to compare queries with context.
- To create a semantic search engine is actually quite simple in the Datasets library.
- First we need to embed all the documents.
- Below we take a small sample from the squad dataset and apply the same embedding logic as before.
- This gives us a new column called embeddings which stores the embeddings of every passage.

``` py
from datasets import load_dataset
squad = load_dataset("squad", split="validation").shuffle(seed=42).select(range(100))
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input["attention_mask"])
squad_with_embeddings = squad.map(
    lambda x: {"embeddings": get_embeddings(x["context"]).cpu().numpy()[0]}
)
```

- We use a special FAISS index for fast nearest neighbour lookup.
- Once we have our embeddings, we need a way to find nearest neighbors for a query.
- The Datasets library provides a special object called FAISS which allows you to quickly compare embedding vectors.
- So we add the FAISS index, embed a question and we found the 3 most similar articles which might store the answer.

``` py
squad_with_embeddings.add_faiss_index(column="embeddings")
question = "Who headlined the halftime show for Super Bowl 50?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
scores, samples = squad_with_embeddings.get_nearest_examples(
    "embeddings", question_embedding, k=3
)
```
