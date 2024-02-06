# Byte-Pair Encoding tokenization

- Byte-Pair algorithm was initially proposed as a text compression algorithm
- But it is also very well suited as a tokenizer for language models.
- The idea of BPE is to divide words into the sequence of 2 words units.
- BPE training is done on a standardized and pre-tokenized corpus.
- BPE training starts with an initial vocabulary and increses it to the desired size.
- To tokenize a text, it is sufficient to divide it into elementary units and then apply the merging rules successively.

