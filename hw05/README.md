## BIOS740 - HW05 Transformers

In this assignment, you will implement Transformers model step-by-step by referencing the original paper, (https://arxiv.org/pdf/1706.03762.pdf). You will also use a toy dataset to solve a vector-to-vector problem which is a subset of sequence-to-sequence problem.

This assignment consists of four parts, with detailed instructions and testing functions provided in `Transformers.ipynb`. The implementation should be done by modifying the necessary functions and classes in `transformers.py`. Follow the notebook's guidelines carefully to ensure correctness.

In this notebook, you will learn how to implement an Encoder-Decoder based Transformers in a step-by-step manner. We will implement a simpler version here, where the simplicity arise from the task that we are solving, which is a vector-to-vector task. This essentially means that the length of input and output sequence is **fixed** and we dont have to worry about variable length of sequences. This makes the implementation simpler.

1. Part I (Preparation): We will preprocess a toy dataset that consists of input arithmetic expression and an output result of the expression
2. Part II (Implement Transformer blocks): we will look how to implement building blocks of a Transformer. It will consist of following blocks
   - MultiHeadAttention
   - FeedForward
   - LayerNorm
   - Encoder Block
   - Decoder Block
3. Part III (Data Loading): We will use the preprocessing functions in part I and the positional encoding module to construct the Dataloader.
4. Part IV (Train a model): In the last part we will look at how to fit the implemented Transformer model to the toy dataset.

