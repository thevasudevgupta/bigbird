# BigBird

[WIP] Lemme finish this first ...

## Introduction

`BigBird` is the transformer based model which is relying on `block sparse attention` instead of normal attention (which can be found in BERT). It can handle sequence length `upto 4096` at a very low compute cost compared to BERT on that long sequences. It has achieved SOTA on various tasks involving very long sequences (typically > 1024) such as long documents summarization, question-answering with longer contexts.

## Why long range attention

Tasks such as question-answering, summarization needs model to understand the global information to be able to perform nicely. 

For instance, for question-answering model needs to capture information about question and part of context to be able to answer, which generates the need to attend question everytime it attends part of context. And then global attention comes into picture.

## Big Bird block sparse attention

Paper suggested to do attention over `global tokens`, `sliding tokens`, & `random tokens` instead of doing it over complete sequence when sequence length is very large (>1024) as compute cost increase significantly in that case. Theoretically, compute complexity reduced to `n` from `n^2` this way.

Authors hardcoded attention matrix and used a simple (but cool) trick to speed up training/inference process on gpu/tpu.

![BigBird block sparse attention](assets/block_sparse.png)
*Note: on the top, we have 2 extra sentences. If you notice, every token is just switched by one place in both sentence. This is how sliding attention is implemented. When `q[i]` is multiplied with `k[i,0:3]`, we will get sliding attention score for q[i] (where `i` is index of element in sequence).*

Let's try to understand above diagram...

### Global Attention

For global attention, each query is simply attending all the other tokens in sequence & is getting attended by every other token. Checkout `blue` blocks in above figure.

```python
# 1st & last token attends all other tokens
Q[0] x (K[0], K[1], K[2], ......, K[n])
Q[1] x (K[0], K[1], K[2], ......, K[n])

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

**Note:** Current implementation further divides sequence into blocks and computation is performed over each block of tokens instead of over single token for making the whole process more efficient on gpu/tpu.
Hence, `BigBird` implemented in `HuggingFace` is currently having 1st few tokens & last few tokens (depending on block size) as global tokens.

## ITC vs ETC

Further, BigBird model is pretrained using 2 different strategies: `ITC` & `ETC`. `ITC` is simply what we discussed above. While in `ETC`, some more tokens are made global such that they will attend / will get attented by all tokens. Paper claimed that this can lead to increase in performance on several tasks.

|                   | ITC                                   | ETC                                   |
|-------------------|---------------------------------------|---------------------------------------|
| `global_tokens`   | 2 x `block_size`                      | `extra_tokens` + 2 x `block_size`     |
| random_tokens     | `num_random_blocks` x `block_size`    | `num_random_blocks` x `block_size`    |
| sliding_tokens    | 3 x `block_size`                      | 3 x `block_size`                      |

<!-- ### How it is differernt from Longformer attention -->

## Using BigBird with Hugging Face transformers

You can use `BigBird` just like any other model available in `HuggingFace`. Let's see how...

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

It's better to keep following points in mind while working with big bird:

* Sequence length must be a multiple of block size i.e. `seqlen % block_size = 0`
* Current implementation doesn't support `num_random_blocks = 0`
* Currently, `HuggingFace` version doesn't support `ETC` and hence only 1st & last block will be global.

## What's next ?

[@patrickvonplaten](https://github.com/patrickvonplaten) has made a really cool [notebook](https://colab.research.google.com/drive/1BAraNpl98loPKG3NvdjJuCLCfvNOZO28) on how to evaluate `BigBirdForQuestionAnswering` on `trivia-qa` dataset. Feel free to play with big bird using that notebook.

You will soon see `BigBirdPegasus` in the library and will be able to do `long documents summarization`💥 easily.

## End Notes

Original implementation of block sparse attention matrix can be found [here](https://github.com/google-research/bigbird/blob/master/bigbird/core/attention.py). You can find `HuggingFace` version [here](https://github.com/huggingface/transformers/pull/10183).

**Feel free to raise an issue, incase you found something wrong here. Star 🌟 this repo if you found this helpful.**
