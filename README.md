# <p align=center>`Big Bird`</p>
 
[WIP] Lemme finish this first ...

## Introduction

**BigBird** (introduced in [paper](https://arxiv.org/abs/2007.14062)) is the transformer based model which is relying on **block sparse attention** instead of normal attention (which can be found in BERT). It can handle sequence length **upto 4096** at a very low compute cost compared to BERT on that long sequences. It has achieved SOTA on various tasks involving very long sequences (typically > 1024) such as long documents summarization, question-answering with long contexts.

Before going into depth of this article, remember that best results are obtained only with BERT like attention as compared to block sparse attention. But since BERT attention compute requirements scales significantly as sequence length increases, we need some kinda alternative which can approximate BERT attention and get equivalent results at less compute. Other way to think is that if we have $\infty$ compute & $\infty$ time (practially not possible), then BERT attention is better than block sparse attention and will yeild better results.

If you wonder that why we need more compute, (no issues!) we will cover that in later sections.

## Why long range attention

Tasks such as question-answering, summarization needs model to understand the global information to be able to perform nicely. 

For instance, for question-answering model needs to capture information about question and part of context to be able to answer, which generates the need to attend question everytime it attends part of context. And then global attention comes into picture.

## Big Bird block sparse attention

Paper suggested to do attention over **global tokens**, **sliding tokens**, & **random tokens** instead of doing it over complete sequence when sequence length is very large (>1024) as compute cost increase significantly in that case. Theoretically, compute complexity reduced to `n` from `n^2` this way.

Authors hardcoded attention matrix and used a simple (but cool) trick to speed up training/inference process on gpu/tpu.

![BigBird block sparse attention](assets/block_sparse.png)
*Note: on the top, we have 2 extra sentences. If you notice, every token is just switched by one place in both sentence. This is how sliding attention is implemented. When `q[i]` is multiplied with `k[i,0:3]`, we will get sliding attention score for `q[i]` (where `i` is index of element in sequence).*

You can find the implementation of `block_sparse` attention starting from [here](https://github.com/vasudevgupta7/transformers/blob/5f2d6a0c93ca2017961199aa04a344b9b779d454/src/transformers/models/big_bird/modeling_big_bird.py#L513). Have a look, this may look very scary 😨😨 now. But this article will surely ease your life in understanding the code.

### Global Attention

For global attention, each query is simply attending all the other tokens in sequence & is getting attended by every other token. Let's assume `Vasudev` (1st token) & `them` (last token) to be global. You can clearly see that these tokens are involved in all the attention computation (blue boxes).

```python
# pseudo code

# 1st & last token attends all other tokens
Q[0] x (K[0], K[1], K[2], ......, K[n])
Q[n] x (K[0], K[1], K[2], ......, K[n])

# 1st & last token getting attended by all other tokens
K[0] x (Q[0], Q[1], Q[2], ......, Q[n])
K[n] x (Q[0], Q[1], Q[2], ......, Q[n])
```

### Sliding Attention

Key sequence is copied 3 times with each element shifted to right in one of the copy & to the left in the other copy. Now if we multiply query sequence vectors by these 3 sequences vectors, we will cover all the sliding tokens. Compute capacity of that
will be only `O(3xn)` or simpy `O(n)`. Refer below figure for the clear idea. You can clearly see 3 sequences in the top of figure with 2 of them switched by one token.

```python
# what we want to do
Q[i] x (K[i-1], K[i], K[i+1])

# efficient implementation in code (assume element-wise multiplication 👇)
(Q[0], Q[1], Q[2], ......, Q[n-1], Q[n]) x (K[1], K[2], K[3], ......, K[n], K[0])
(Q[0], Q[1], Q[2], ......, Q[n]) x (K[n], K[1], K[2], ......, K[n-1])
(Q[0], Q[1], Q[2], ......, Q[n]) x (K[0], K[1], K[2], ......, K[n])
```

**Note:** Now, each sequence is getting mutiplied by only 3 sequences to keep `window_size = 3` and we are able to reduce the time complexity.

### Random Attention

Random attention is ensuring that each query token will attend few random tokens as well.

```python
# r1, r2, r are some random indices; Note: r1, r2, r3 are different for each row 👇
Q[0] x (Q[r1], Q[r2], ......, Q[r])
Q[1] x (Q[r1], Q[r2], ......, Q[r])
.
.
.
Q[n] x (Q[r1], Q[r2], ......, Q[r])
```

<!-- **Note:** Current implementation further divides sequence into blocks and computation is performed over each block of tokens instead of over single token for making the whole process more efficient on gpu/tpu. -->
Hence, BigBird implemented in HuggingFace is currently having 1st few tokens & last few tokens (depending on block size) as global tokens.

### Graphical view

<img src="assets/global.png" width=256 height=200> </img>
<img src="assets/sliding.png" width=256 height=200> </img>
<img src="assets/random.png" width=256 height=200> </img> <br>
*Above figure shows `global`, `sliding` & `random` connections respectively with the help of graph, where each node corresponds to token and each line represents attention score. If no connection is made between 2 tokens, then attention score is assumed to be 0.*

Now if we want to share information between two nodes (or tokens), we need to travel across various nodes in the path. Since all the nodes are not connected in a single layer, we may need multiple layers to capture the information which normal attention is capturing in a single layer.

In case of normal attention, all 81 connections would have been present in above figure & it would be easy for model can transfer information from one token to other in a single layer.

| Attention Type  | `global_blocks`   | `sliding_blocks` | `random_blocks`     |
|-----------------|-------------------|------------------|---------------------|
| `original_full` | `n // block_size` | 0                | 0                   |
| `block_sparse`  | 2                 | 3                | `num_random_blocks` |

In normal attention, each token is queried over every other token and information is flowed among all the tokens in a single layer.
While in block sparse attention, single layer won't be able to capture complete sequence information since all tokens are not attending all other tokens (though global tokens is trying to do this).

Note that since for most of the tasks we need to capture complete global information, block sparse attention may need more no of layers to be able to perform similar to normal attention. Now, this can introduce time complexity of O(n^2) if we need as many layers as sequence length. But BigBird paper claimed that if we have good number of global tokens/ random tokens, we can actually get equivalent results.

Other way to think is that normal attention simply means all tokens are global.

## Time complexity

| Attention Type  | Sequence length | Time Complexity |
|-----------------|-----------------|-----------------|
| `original_full` | 512             | `O(n^2)`        |
|                 | 1024            | 4 x `O(n^2)`    |
|                 | 4096            | 64 x `O(n^2)`   |
| `block_sparse`  | 1024            | 2 x `O(n^2)`    |
|                 | 4096            | 8 x `O(n^2)`    |

<details>

<summary>Expand this snippet in case you wanna see the calculations</summary>

```md
BigBird time complexity = O(w x n + r x n + g x n)
BERT time complexity = O(n^2)

Assumptions:
    w = 3 x 64
    r = 3 x 64
    g = 2 x 64

When seqlen = 512
=> **time complexity in BERT = 512^2**

When seqlen = 1024
=> time complexity in BERT = (2 x 512)^2
=> **time complexity in BERT = 4 x 512^2**

=> time complexity in BigBird = (8 x 64) x (2 x 512)
=> **time complexity in BigBird = 2 x 512^2**

When seqlen = 4096
=> time complexity in BERT = (8 x 512)^2
=> **time complexity in BERT = 64 x 512^2**

=> compute in BigBird = (8 x 64) x (8 x 512)
=> compute in BigBird = 8 x (512 x 512)
=> **time complexity in BigBird = 8 x 512^2**
```

</details>

## ITC vs ETC

BigBird model is finetuned using 2 different strategies: **ITC** & **ETC**. ITC (internal transformer construction) is simply what we discussed above. While in ETC (extended transformer construction), some extra tokens are made global such that they will attend / will get attented by all tokens.

|                   | ITC                                   | ETC                                   |
|-------------------|---------------------------------------|---------------------------------------|
| Attention Matrix  |<a href="https://www.codecogs.com/eqnedit.php?latex=A&space;=&space;\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A&space;=&space;\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" title="A = \begin{bmatrix} 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & & & & & & 1 \\ 1 & & & & & & 1 \\ 1 & & & & & & 1 \\ 1 & & & & & & 1 \\ 1 & & & & & & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{bmatrix}" /></a> | <a href="https://www.codecogs.com/eqnedit.php?latex=B&space;=&space;\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?B&space;=&space;\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" title="B = \begin{bmatrix} 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{bmatrix}" /></a> |
| `global_tokens`   | 2 x `block_size`                      | `extra_tokens` + 2 x `block_size`     |
| `random_tokens`   | `num_random_blocks` x `block_size`    | `num_random_blocks` x `block_size`    |
| `sliding_tokens`  | 3 x `block_size`                      | 3 x `block_size`                      |
| Benefits          | Lesser compute                        | Performs better on tasks that needs more global tokens such as `question-answering` (complete question is required to query over the context) |

## BigBird vs Longformer

### Normal sparse vs Block sparse

longformer

## Using BigBird with Hugging Face transformers

You can use BigBird just like any other model available in HuggingFace. Let's see how...

```python
from transformers import BigBirdModel

# loading bigbird from its pretrained checkpoint
model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
# This will init the model with default configuration i.e. attention_type = "block_sparse" num_random_blocks = 3, block_size = 64.
# But You can freely change these arguments with any checkpoint. These 3 argument will just change the number of tokens each query token is going to attend.
model = BigBirdModel.from_pretrained("google/bigbird-roberta-base", num_random_blocks=2, block_size=16)

# By setting attention_type to `original_full`, BigBird will be relying on full attention of n^2 complexity. This way BigBird is 99.9 % similar to BERT.
model = BigBirdModel.from_pretrained("google/bigbird-roberta-base", attention_type="original_full")
```

There are total 3 checkpoints available in huggingface_hub (at the point of writing this article): [`bigbird-roberta-base`](https://huggingface.co/google/bigbird-roberta-base), [`bigbird-roberta-large`](https://huggingface.co/google/bigbird-roberta-large), [`bigbird-base-trivia-itc`](https://huggingface.co/google/bigbird-base-trivia-itc). First 2 checkpoints comes from pretraining `BigBirdForPretraining` with `masked_lm loss`; while the last one corresponds to the checkpoint after finetuning `BigBirdForQuestionAnswering` on `trivia-qa` dataset.

It's important to keep following points in mind while working with big bird:

* Sequence length must be a multiple of block size i.e. `seqlen % block_size = 0`
* Current implementation doesn't support `num_random_blocks = 0`
* Currently, HuggingFace version **doesn't support ETC** and hence only 1st & last block will be global.
* When using big bird as decoder (or using `BigBirdForCasualLM`), `attention_type` should be `original_full`. But you need not worry, 🤗 implementation will automatically switch `attention_type` to `original_full` incase you forget to do that.

## What's next?

[@patrickvonplaten](https://github.com/patrickvonplaten) has made a really cool [notebook](https://colab.research.google.com/drive/1BAraNpl98loPKG3NvdjJuCLCfvNOZO28) on how to evaluate `BigBirdForQuestionAnswering` on `trivia-qa` dataset. Feel free to play with big bird using that notebook.

You will soon see `BigBirdPegasus` in the library and will be able to do **long documents summarization**💥 easily.

## End Notes

Original implementation of **block sparse attention matrix** can be found [here](https://github.com/google-research/bigbird/blob/master/bigbird/core/attention.py). You can find **HuggingFace** version [here](https://github.com/huggingface/transformers/pull/10183).

**Feel free to raise an issue, incase you found something wrong here. Star 🌟 this repo if you found this helpful.**
