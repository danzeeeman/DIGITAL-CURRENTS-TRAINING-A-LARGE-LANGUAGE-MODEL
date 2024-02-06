!!! note "Pipeline Function"

    The pipeline function returns an end-to-end object that performs an NLP task on one or several texts.

    - It includes steps:

        * Pre-Processing
        * Model
        * Post-Processing

``` py
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```
By default, this pipeline selects a particular pretrained model that has been fine-tuned for sentiment analysis in English. The model is downloaded and cached when you create the classifier object. If you rerun the command, the cached model will be used instead and there is no need to download the model again.

- Available pipelines are:

    * feature-extraction (get the vector representation of a text)
    * fill-mask: to fill in the blanks
    * ner (named entity recognition): identifies persons, locations, or organizations
    * question-answering: answers questions using information from a given context
    * sentiment-analysis
    * summarization: reducing a text into a shorter text while keeping almost of the important aspects referenced in the text
    * text-generation
    * translation
    * zero-shot-classification: classify on the basis of listed classes/labels.

- Transformer Models:

    * June 2018: [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
    * October 2018: [BERT](https://arxiv.org/abs/1810.04805)
    * February 2019: [GPT-2](https://arxiv.org/abs/1910.01108)
    * October 2019: [DistilBERT](https://arxiv.org/abs/1910.01108), a distilled version of BERT that is 60% faster, 40% lighter in memory, and still retains 97% of BERT’s performance.
    * October 2019: [BART](https://arxiv.org/abs/1910.13461) and [T5](https://arxiv.org/abs/1910.10683): Both Encoder and Decoder
    * May 2020: [GPT-3](https://arxiv.org/abs/2005.14165): an even bigger version of GPT-2 that is able to perform well on a variety of tasks without the need for fine-tuning (called zero-shot learning)


- Transformer Model Categories:

    * GPT-like (also called auto-regressive Transformer models)
    * BERT-like (also called auto-encoding Transformer models)
    * BART/T5-like (also called sequence-to-sequence Transformer models)


- Transformers are:

    * Language models: trained on large amounts of raw text in a self-supervised fashion
    * Self-supervised learner: here objective is automatically computed from the inputs of the model
    * pretrained model goes through transfer learning, ie, fine-tuned in supervised way using human-annotated labels on given task.
    * Casual Language Modeling: output depends on the past and present inputs, but not the future ones
    * Masked language modeling: predicts a masked word in the sentence


- Transfer Learning

    * The act of initializing a model with another model's weights.
    * Training from scratch requires more data and more compute to achieve comparable results.
    * In NLP, predicting the next word is a common pretraining objective.(GPT)
    * Another common pretraining objective in text is to gues the value of randomly masked words.(BERT)
    * Usually, Transfer Learning is applied by dropping the head of the pretrained model while keeping its body.
    * The pretrained model helps by transferring its knowledge but it also transfers the bias it may contain.
    * OpenAI studied the bias predictions of its GPT-3 model.

|||
-|-  
pre-training | fine-tuning
training from scratch | transfer learning

- Architectures vs. checkpoints

    * Architecture: This is the skeleton of the model — the definition of each layer and each operation that happens within the model.
    * Checkpoints: These are the weights that will be loaded in a given architecture.
    * Model: This is an umbrella term that isn’t as precise as “architecture” or “checkpoint”: it can mean both. This course will specify architecture or checkpoint when it matters to reduce ambiguity.

### Bert Family: 
ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa

### Decoder family: 
CTRL, GPT, GPT-2, Transformer XL, GPT Neo

### Encoder-decoder(sequence-to-sequence) models:
BART, mBART, Marian, T5, Pegasus, ProphetNet, M2M100

