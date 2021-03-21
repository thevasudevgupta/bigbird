# <p align=center>`Big Bird`</p>

[WIP] Lemme finish this first ...

## Introduction

Transformers based models are proving very useful for most of the NLP tasks. Major limitation of transformers is `O(n^2)` time & memory complexity (where n is sequence length). Hence, it's hard to use transformers for very long sequences (generally > 512). Several recent papers tried to focus on this by approximating the full attention matrix. Some of these ideas include Longformer, performer, reformer, clustered attention. You can check the 🤗 recent blog [post](https://huggingface.co/blog/long-range-transformers) in case you are unfamilier with some of these models.

BigBird (introduced in [paper](https://arxiv.org/abs/2007.14062)) is one of the most recent model which is addressing this issue and extending the work of `longformer`. It is relying on **block sparse attention** instead of normal attention (which can be found in BERT). It can handle sequence length **upto 4096** at a very low compute cost compared to BERT on that long sequences. It has achieved SOTA on various tasks involving very long sequences such as long documents summarization, question-answering with long contexts.

Before going into depth of this article, remember that best results are obtained only with BERT like attention as compared to any other approximation of attention matrix. But since BERT attention compute & memory requirements scales quadratically as sequence length increases, we need some kinda alternative which can approximate BERT attention and get equivalent results at less compute. Other way to think is that if we have $\infty$ compute & $\infty$ time (practially not possible), then BERT attention is better than block sparse attention (which we are going to discuss in this post) and will yeild better results.

If you wonder why we need more compute when working with longer sequences, (no worries!) just continue reading this post.

---

Some of the main question when working with normal attention matrix are following:

* Do all the tokens need to attend every other token or just few tokens?
* Why not compute attention only over those tokens that are important?
* How to decide what tokens are important?

---

Keeping these questions in mind, we will try to proceed. But before that let's go over few other questions.

### Why long range attention

```python
# Let's consider a `set` and fill up the tokens of our interest which we should attend.
key_tokens = set()
```

Nearby tokens are important obviously because in a sentence (sequence of words), current word is highly dependent on few future tokens & few past tokens. This idea introduced the concept of `sliding attention`.

```python
# Let's update `set` with nearby tokens
key_tokens.update(sliding_tokens)
```

Long range relationships needs to be captured for lots of tasks. Eg: `question-answering` where model needs to capture information about entire question and most of context to be able to answer correctly.

there are 2 ways of doing this:

* Introduce some tokens which will attend every token and gets attented by all the tokens. Eg: "HuggingFace is building nice libraries for easy NLP". Now, let's say 'building' is global token, then if we want to associate 'NLP' with 'HuggingFace'; 'building' representation will possibly help model to assiciate 'NLP' with 'HuggingFace'.

```python
# fill up global tokens in our `set`
key_tokens.update(global_tokens)
```

* Introduce some random tokens which will transfer information by transfering to other tokens which in turn can transfer to other tokens. This will reduce the cost of information travel from one token to other. This is why `random` attention attention is introduced.

```python
# Let's add random tokens to our `set`
key_tokens.update(random_tokens)
```

Now, we just need our token to attend this `set` & possibly it will represent all the tokens nicely. Similar thing we will do for all the queries.

### Understanding with Graphs

Let's try to understand the need of `global`, `sliding` & `random` attention using the graphs.

<img src="assets/global.png" width=256 height=200> </img>
<img src="assets/sliding.png" width=256 height=200> </img>
<img src="assets/random.png" width=256 height=200> </img> <br>
*Above figure shows `global`, `sliding` & `random` connections respectively in graph, where each node corresponds to token and each dotted-line represents attention score. If no connection is made between 2 tokens, then attention score is assumed to 0.*

BigBird block sparse attention is simply combination of these 3 figures. While in normal attention, all 81 connections (note: total 9 nodes are present) would have been present in above figure. You can simply think of normal attention as all the tokens being global.

| Attention Type  | `global_tokens`   | `sliding_tokens` | `random_tokens`                    |
|-----------------|-------------------|------------------|------------------------------------|
| `original_full` | `n`               | 0                | 0                                  |
| `block_sparse`  | 2 x `block_size`  | 3 x `block_size` | `num_random_blocks` x `block_size` |

*`original_full` represents BERT attention while `block_sparse` represents BigBird attention*

**Normal attention:** Model can transfer information from one token to other directly in a single layer, since each token is queried over every other token and information can be flowed among all the tokens in a single layer.

**Block sparse attention:** If we want to share information between two nodes (or tokens), we may need to travel across various other nodes in the path; since all the nodes are not directly connected in a single layer. Hence, we may need multiple layers to capture the entire information of the sequence; which normal attention can capture in a single layer. This can introduce time complexity of `O(n^2)` because now we need as many layers as sequence length.

## Big Bird block sparse attention

Paper suggested to attend only few tokens namely, **global tokens**, **sliding tokens**, & **random tokens** instead of attending the complete sequence when sequence length is very large (typically > 1024) as compute cost increases significantly in that case. Theoretically, this way compute complexity gets reduced to `O(n)` from `O(n^2)`.

Authors hardcoded attention matrix to compute attention over only  and used a cool trick to speed up training/inference process on gpu/tpu.

![BigBird block sparse attention](assets/block_sparse.png)
*Note: on the top, we have 2 extra sentences. If you notice, every token is just switched by one place in both sentence. This is how sliding attention is implemented. When `q[i]` is multiplied with `k[i,0:3]`, we will get sliding attention score for `q[i]` (where `i` is index of element in sequence).*

You can find the implementation of `block_sparse` attention starting from [here](https://github.com/vasudevgupta7/transformers/blob/5f2d6a0c93ca2017961199aa04a344b9b779d454/src/transformers/models/big_bird/modeling_big_bird.py#L513). Have a look, this may look very scary 😨😨 now. But this article will surely ease your life in understanding the code.

### Global Attention

For global attention, each query is simply attending all the other tokens in sequence & is getting attended by every other token. Let's assume `Vasudev` (1st token) & `them` (last token) to be global. You can clearly see that these tokens are involved in all the attention computation (blue boxes).

```python
# pseudo code

# 1st & last token attends all other tokens
Q[0] x (K[0], K[1], K[2], ......, K[n-1])
Q[n-1] x (K[0], K[1], K[2], ......, K[n-1])

# 1st & last token getting attended by all other tokens
K[0] x (Q[0], Q[1], Q[2], ......, Q[n-1])
K[n-1] x (Q[0], Q[1], Q[2], ......, Q[n-1])
```

### Sliding Attention

Key sequence is copied 3 times with each element shifted to right in one of the copy & to the left in the other copy. Now if we multiply query sequence vectors by these 3 sequences vectors, we will cover all the sliding tokens. Compute capacity of that
will be only `O(3xn)` or simpy `O(n)`. Refer below figure for the clear idea. You can clearly see 3 sequences in the top of figure with 2 of them switched by one token.

```python
# what we want to do
Q[i] x (K[i-1], K[i], K[i+1])

# efficient implementation in code (assume element-wise multiplication 👇)
(Q[0], Q[1], Q[2], ......, Q[n-2], Q[n-1]) x (K[1], K[2], K[3], ......, K[n-1], K[0])
(Q[0], Q[1], Q[2], ......, Q[n-1]) x (K[n-1], K[0], K[1], ......, K[n-2])
(Q[0], Q[1], Q[2], ......, Q[n-1]) x (K[0], K[1], K[2], ......, K[n-1])
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
Q[n-1] x (Q[r1], Q[r2], ......, Q[r])
```

**Note:** Current implementation further divides sequence into blocks & each notation is defined w.r.to block instead of token. Hence, BigBird implemented in HuggingFace is currently having 1st block & last block as global tokens.

### Normal sparse vs Block sparse

TODO

## Time & Memory complexity

| Attention Type  | Sequence length | Time & Memory Complexity |
|-----------------|-----------------|--------------------------|
| `original_full` | 512             | `M`                      |
|                 | 1024            | 4 x `M`                  |
|                 | 4096            | 64 x `M`                 |
| `block_sparse`  | 1024            | 2 x `M`                  |
|                 | 4096            | 8 x `M`                  |

*In this table, I am trying to compare time & space complexity of BERT attention and BigBird block sparse attention.*

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

ITC requires less compute since very few tokens are globals & model can still capture global information with them. On the other hand, ETC can be very helpful for the tasks in which we need lot of global tokens such as `question-answering` in which entire question should be global, with some tokens of context to be able to understand context; `summarization` since model needs to understand the overall context of very long paragraph.

Below table tries to summarize ITC & ETC:

|                                              | ITC                                   | ETC                                  |
|----------------------------------------------|---------------------------------------|--------------------------------------|
| Attention Matrix with global attention       |<a href="https://www.codecogs.com/eqnedit.php?latex=A&space;=&space;\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A&space;=&space;\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" title="A = \begin{bmatrix} 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & & & & & & 1 \\ 1 & & & & & & 1 \\ 1 & & & & & & 1 \\ 1 & & & & & & 1 \\ 1 & & & & & & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{bmatrix}" /></a> | <a href="https://www.codecogs.com/eqnedit.php?latex=B&space;=&space;\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?B&space;=&space;\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" title="B = \begin{bmatrix} 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{bmatrix}" /></a> |
| `global_tokens`   | 2 x `block_size`                      | `extra_tokens` + 2 x `block_size`     |
| `random_tokens`   | `num_random_blocks` x `block_size`    | `num_random_blocks` x `block_size`    |
| `sliding_tokens`  | 3 x `block_size`                      | 3 x `block_size`                      |

## Using BigBird with Hugging Face transformers

You can use `BigBirdModel` just like any other model available in HuggingFace. Let's see some code below:

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

* Sequence length must be a multiple of block size i.e. `seqlen % block_size = 0`. You need not worry since 🤗 implementation will automatically `[PAD]` (to smallest multiple of block size which is greater than sequence length) if batch sequence length is not a multiple of `block_size`.
* Current implementation doesn't support `num_random_blocks = 0`.
* Currently, HuggingFace version **doesn't support ETC** and hence only 1st & last block will be global.
* When using big bird as decoder (or using `BigBirdForCasualLM`), `attention_type` should be `original_full`. But you need not worry, 🤗 implementation will automatically switch `attention_type` to `original_full` incase you forget to do that.

## What's next?

[@patrickvonplaten](https://github.com/patrickvonplaten) has made a really cool [notebook](https://colab.research.google.com/drive/1BAraNpl98loPKG3NvdjJuCLCfvNOZO28) on how to evaluate `BigBirdForQuestionAnswering` on `trivia-qa` dataset. Feel free to play with big bird using that notebook.

You will soon see `BigBirdPegasus` in the library and will be able to do **long documents summarization**💥 easily. Meanwhile, I created another [notebook](https://colab.research.google.com/github/vasudevgupta7/bigbird-intuition/blob/main/notebooks/bigbird_narrativeqa.ipynb) for you. In case you wanna know how to fine-tune bigbird roberta on abstractive question-answering, feel free to check it out.

## End Notes

Original implementation of **block sparse attention matrix** can be found [here](https://github.com/google-research/bigbird/blob/master/bigbird/core/attention.py). You can find 🤗 version [here](https://github.com/huggingface/transformers/pull/10183).

**Feel free to raise an issue, incase you found something wrong here. Star 🌟 this repo if you found this helpful.**
