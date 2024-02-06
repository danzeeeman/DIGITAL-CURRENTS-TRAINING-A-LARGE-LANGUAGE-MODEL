# HuggingFace Tokenizers Library

## Introduction

- We will learn:
    - How to train a new tokenizer similar to the one used by a given checkpoint on a new corpus of texts
    - The special features of fast tokenizers
    - The differences between the three main subword tokenization algorithms used in NLP today
    - How to build a tokenizer from scratch with the HuggingFace Tokenizers library and train it on some data

- After this deep dive into tokenizers, you should:
    - Be able to train a new tokenizer using an old one as a template
    - Understand how to use offsets to map tokens’ positions to their original span of text
    - Know the differences between BPE, WordPiece, and Unigram
    - Be able to mix and match the blocks provided by the 🤗 Tokenizers library to build your own tokenizer
    - Be able to use that tokenizer inside the 🤗 Transformers library

