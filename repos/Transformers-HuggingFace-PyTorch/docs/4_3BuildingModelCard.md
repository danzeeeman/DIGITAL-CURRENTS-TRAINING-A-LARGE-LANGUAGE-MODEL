# Building a Model Card

- It is the central definition of the model, ensuring reusability by fellow community members and reproducibility of results, and providing a platform on which other members may build their artifacts.
- Creating the model card is done through the README.md file, which is a Markdown file.
- The model card usually starts with a very brief, high-level overview of what the model is for, followed by additional details in the following sections:
    - Model description
    - Intended uses & limitations
    - How to use
    - Limitations and bias
    - Training data
    - Training procedure
    - Evaluation results

## Model description

- The model description provides basic details about the model. 
- This includes the architecture, version, if it was introduced in a paper, if an original implementation is available, the author, and general information about the model. 
- Any copyright should be attributed here. 
- General information about training procedures, parameters, and important disclaimers can also be mentioned in this section.

## Intended uses & limitations

- Here you describe the use cases the model is intended for, including the languages, fields, and domains where it can be applied. 
- This section of the model card can also document areas that are known to be out of scope for the model, or where it is likely to perform suboptimally.

## How to use

- This section should include some examples of how to use the model. 
- This can showcase usage of the pipeline() function, usage of the model and tokenizer classes, and any other code you think might be helpful.

## Training data

- This part should indicate which dataset(s) the model was trained on. 
- A brief description of the dataset(s) is also welcome.

## Training procedure

- In this section you should describe all the relevant aspects of training that are useful from a reproducibility perspective. 
- This includes any preprocessing and postprocessing that were done on the data, as well as details such as the number of epochs the model was trained for, the batch size, the learning rate, and so on.

## Variable and metrics

- Here you should describe the metrics you use for evaluation, and the different factors you are mesuring. 
- Mentioning which metric(s) were used, on which dataset and which dataset split, makes it easy to compare you model’s performance compared to that of other models. 
- These should be informed by the previous sections, such as the intended users and use cases.

## Evaluation results

- Finally, provide an indication of how well the model performs on the evaluation dataset.
- If the model uses a decision threshold, either provide the decision threshold used in the evaluation, or provide details on evaluation at different thresholds for the intended uses.

- Model cards are not a requirement when publishing models, and you don’t need to include all of the sections described above when you make one. 
- However, explicit documentation of the model can only benefit future users

# Model card metadata

``` bash
---
language: fr
license: mit
datasets:
- oscar
---
```

- This metadata is parsed by the Hugging Face Hub, which then identifies this model as being a French model, with an MIT license, trained on the Oscar dataset.

