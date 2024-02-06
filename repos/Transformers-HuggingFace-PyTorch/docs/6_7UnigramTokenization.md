# Unigram tokenization

- The overall training strategy is to start with a very large vocabulary and then iteratively reduce it.
- Unigram model is a type of Statistical Language Model assuming that the occurrence of each word is independent of its previous word.
- Let's look at a toy example to understand how to train a Unigram LM tokenizer and how to use it to tokenize a new text.
- 1st iteration:
    - E-step: Estimate the probabilities
- Additional explanations: 
    - How do we tokenize a text with Unigram LM? 
    - How do we calculate the loss on the training corpus?
- 1st iteration:
    - M-step: Remove the token that least impacts the loss on the corpus.
- 2nd iteration:
    - E-step: Estimate the probabilities
- 2nd iteration:
    - M-step: Remove the token that least impacts the loss on the corpus.
- In practice, when we want to find the optimal tokenization of a word according to a Unigram model, we use the Viterbi algorithm instead of listing and calculating and comparing all the possibilities.
