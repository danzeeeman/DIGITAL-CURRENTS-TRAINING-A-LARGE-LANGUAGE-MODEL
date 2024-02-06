# The Hugging Face Hub

- We will focus on the models in this section, and take a look at the datasets in the next section.

## Navigating the Model Hub

- Huggingface landing page: https://huggingface.co/
- To access models: https://huggingface.co/models
- Left side has different categories: Tasks, framework, language, datasets, etc
- Selecting model shows its model card.
- The model card contains information about the model, its description, intended use, limitations and biases.
- It can also show code snippets on how to use the model as well as relevant information, training procedure, data processing, evaluation results or copyrights.
- On right of the model card is the inference api, it can be used to play with the model directly.
- At the top of the model card lies the model tags. These includes the model task as well as any other tag that is relevant to the categories on the left of the model screen.
- The files and vsersions tab displays the architecture of the repository of that model. Here, we can see all the files that define this model.
- You will see all usual features of a git repository, branches available, commit history as well as commit diff.

- 3 different buttons are available at the top of the model card:
    - Use Accelerated Inference: the first one shows how to use the inference api programmatically
    - Use in SageMaker: the second one shows how to train this model in SageMaker
    - Use in Transformers: the last one shows how to load that model within the appropriate library
