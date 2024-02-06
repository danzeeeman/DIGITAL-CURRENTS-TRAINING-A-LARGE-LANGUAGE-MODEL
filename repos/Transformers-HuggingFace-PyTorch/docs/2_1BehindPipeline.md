# Behind the pipeline

## Preprocessing with a tokenizer
    -> All preprocessing needs to be done in exactly the same way as when the model was pretrained.
    -> To do this, we use the AutoTokenizer class and its from_pretrained() method.
    ->  Using the checkpoint name of our model, it will automatically fetch the data associated with the model’s tokenizer and cache it.

``` py
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

    -> Next step is to convert the list of input IDs to tensors.
    -> To specify the type of tensors we want to get back (PyTorch, TensorFlow, or plain NumPy), we use the return_tensors argument:

``` py
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") #return_tensors: To specify the type of tensors we want to get back 
# truncation=True, Any sentence longer than the maximum the model can handle is truncated
print(inputs)
```

    -> The output itself is a dictionary containing two keys, input_ids and attention_mask. 
    -> input_ids contains two rows of integers (one for each sentence) that are the unique identifiers of the tokens in each sentence.
    -> attention mask indicates where padding has been applied, so the model does not pay attention to it.

## Going through the model
    -> Pretrained model can be downloaded similarly as tokenizer using AutoModel class.
    -> AutoModel class loads a model without its pretraining head.

``` py
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint) # downloads configuration of the model as well as pre-trained weights, only initantiates the body of the model
# downloaded same checkpoint used in pipeline(cached already) and instantiated a model with it
```

    -> This architecture contains only the base Transformer module: given some inputs, it outputs what we’ll call hidden states, also known as features. For each model input, we’ll retrieve a high-dimensional vector representing the contextual understanding of that input by the Transformer model.
    -> While these hidden states can be useful on their own, they’re usually inputs to another part of the model, known as the head. Different NLP tasks could have been performed with the same architecture, but each of these tasks will have a different head associated with it.

### A high-dimensional vector?
- The vector output by the Transformer module is usually large. It generally has three dimensions:
    * Batch size: The number of sequences processed at a time (2 in our example).
    * Sequence length: The length of the numerical representation of the sequence (16 in our example).
    * Hidden size: The vector dimension of each model input.

- It is said to be “high dimensional” because of the last value. The hidden size can be very large (768 is common for smaller models, and in larger models this can reach 3072 or more).

``` py
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
# torch.Size([2, 16, 768])
```

- Note that the outputs of Transformers models behave like namedtuples or dictionaries. You can access the elements by attributes (like we did) or by key (outputs["last_hidden_state"]), or even by index if you know exactly where the thing you are looking for is (outputs[0]).

### Model heads: Making sense out of numbers
- The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension.
- The output of the Transformer model is sent directly to the model head to be processed.
- There are many different architectures available in Transformers, with each one designed around tackling a specific task:
    - *Model (retrieve the hidden states)
    - *ForCausalLM
    - *ForMaskedLM
    - *ForMultipleChoice
    - *ForQuestionAnswering
    - *ForSequenceClassification
    - *ForTokenClassification, etc

- For our example, we will need a model with a sequence classification head (to be able to classify the sentences as positive or negative). So, we won’t actually use the AutoModel class, but AutoModelForSequenceClassification:

``` py
from transformers import AutoModelForSequenceClassification
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)
# torch.Size([2, 2])
```
- Now if we look at the shape of our inputs, the dimensionality will be much lower: the model head takes as input the high-dimensional vectors we saw before, and outputs vectors containing two values (one per label):
    - Since we have just two sentences and two labels, the result we get from our model is of shape 2 x 2.

## Postprocessing the output

``` py
print(outputs.logits)
```

- logits are raw, unnormalized scores outputted by the last layer of the model. - To be converted to probabilities, they need to go through a SoftMax layer (all Transformers models output the logits, as the loss function for training will generally fuse the last activation function, such as SoftMax, with the actual loss function, such as cross entropy)

``` py
import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

- It returns probability scores.
- To get the labels corresponding to each position, we can inspect the id2label attribute of the model config.

``` py
model.config.id2label
# {0: 'NEGATIVE', 1: 'POSITIVE'}
```
