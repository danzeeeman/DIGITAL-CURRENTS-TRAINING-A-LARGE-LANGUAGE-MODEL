# Instantiate a Transformers model

- AutoModel API allows you to instantiate a pretrained model from any checkpoint.

``` py
from transformers import AutoModel

bert_model = AutoModel.from_pretrained("bert-base-cased")
print(type(bert_model))

gpt_model = AutoModel.from_pretrained("gpt2")
print(type(gpt_model))

bart_model = AutoModel.from_pretrained("facebook/bart-base")
print(type(bart_model))
```

- It will pick the right model class from the library to instantiate the proper architecture and loads the weights of the pre-trained model inside.

## Checkpoint or local folder

- Behind the AutoModel.from_pretrained() method:
    - config.json file: attributes necessary to build the model architecture. This file also contains some metadata, such as where the checkpoint originated and what Transformers version you were using when you last saved the checkpoint.
    - pytorch_model.bin file: known as the state dictionary; it contains all your model’s weights
    - configuration is necessary to know your model’s architecture, while the model weights are your model’s parameters.
    - The weights can be downloaded and cached (so future calls to the from_pretrained() method won’t re-download them) in the cache folder, which defaults to ~/.cache/huggingface/transformers. You can customize your cache folder by setting the HF_HOME environment variable.
    - To instantiate a pre-trained model, the AutoConfig API will first check in config file to look at the config class that should be used. The config class depends on the type of model(bert, gpt-2, etc). Once it attaches a proper config class, it can instantiate that configuration which is a blueprint to know how to create a model. It uses this configuration class to find the proper model which is combined to the logit configuration to load the model. this model is not yet trained model as it just being initialized with the random weights.
    - The last step is to loads weight from the model file inside this model(above loaded model).
    
- The AutoConfig API allows you to instantiate the configuration of a pretrained model from any checkpoint:

``` py
from transformers import AutoConfig

bert_config = AutoConfig.from_pretrained("bert-base-cased")
print(type(bert_config))

gpt_config = AutoConfig.from_pretrained("gpt2")
print(type(gpt_config))

bart_config = AutoConfig.from_pretrained("facebook/bart-base")
print(type(bart_config))
```

- But you can also use the specific class if you know it:

``` py
from transformers import BertConfig

bert_config = BertConfig.from_pretrained("bert-base-cased")
print(type(bert_config))

from transformers import GPT2Config

gpt_config = GPT2Config.from_pretrained("gpt2")
print(type(gpt_config))

from transformers import BartConfig

bart_config = BartConfig.from_pretrained("facebook/bart-base")
print(type(bart_config))
```

- The configuration(bert_config, gpt_config, bart_config) contains all the information needed to load the model/create the model architecture.
``` py
from transformers import BertConfig

bert_config = BertConfig.from_pretrained("bert-base-cased")
print(bert_config)
```

- Then you can instantiate a given model with random weights from this config. ie, once we have the configuration we can create a model which has same architecture as the checkpoint which is from it was initialized. We can train it from scratch.
- We can also change any part of its configurations using keyword arguments

``` py
# Same architecture as bert-base-cased
from transformers import BertConfig, BertModel

bert_config = BertConfig.from_pretrained("bert-base-cased")
bert_model = BertModel(bert_config)

# Using only 10 layers instead of 12
from transformers import BertConfig, BertModel

bert_config = BertConfig.from_pretrained("bert-base-cased", num_hidden_layers=10)
bert_model = BertModel(bert_config)
```

## Saving a model

- To save a model, we just have to use the the save_pretrained method.

``` py
from transformers import BertConfig, BertModel

bert_config = BertConfig.from_pretrained("bert-base-cased")
bert_model = BertModel(bert_config)

# Training code
bert_model.save_pretrained("my-bert-model")
```

## Reloading a saved model

``` py
from transformers import BertModel

bert_model = BertModel.from_pretrained("my-bert-model")
```

