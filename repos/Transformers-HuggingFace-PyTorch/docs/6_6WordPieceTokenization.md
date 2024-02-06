# WordPiece tokenization

- WordPiece is the tokenization algorithm Google developed to pretrain BERT. 
- It has since been reused in quite a few Transformer models based on BERT, such as DistilBERT, MobileBERT, Funnel Transformers, and MPNET.
- The learning strategy for a WordPiece tokenizer is similar to that of BPE but differs in the way the score for each candidate token is calculated.

``` bash
score=(freq_of_pair)/(freq_of_first_element X freq_of_second_element)
```

- To tokenize a text with a learned WordPiece tokenizer we look for the longest token present at the beginning of the text.
