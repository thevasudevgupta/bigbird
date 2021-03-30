# BigBird

[WIP] Lemme finish this first ...

## Introduction

Transformer-based models have shown to be very useful for many NLP tasks. However, a major limitation of transformers-based models is it's $O(n^2)$ time & memory complexity (where $n$ is sequence length). Hence, it's computationally very expensive to apply transformer-based models on long sequences ($n > 512$). Several recent papers, *e.g.* `Longformer`, `Performer`, `Reformer`, `Clustered attention`, try to remedy this problem by approximating the full attention matrix. You can checkout 🤗's recent blog [post](https://huggingface.co/blog/long-range-transformers) in case you are unfamilier with these models.

`BigBird` (introduced in [paper](https://arxiv.org/abs/2007.14062)) is one of such recent models to address this issue. `BigBird` relies on **block sparse attention** instead of normal attention (*i.e.* BERT's attention) and can handle sequences up to a length of **4096** at a much lower compute cost compared to BERT. It has achieved SOTA on various tasks involving very long sequences such as long documents summarization, question-answering with long contexts.

**BigBird RoBERTa-like** model is now available in 🤗Transformers. The goal of this post is to give reader an **in-depth** understanding of big bird implementation & ease one's life in using big bird with 🤗Transformers. But, before going into more depth, it is important to remember that the `BigBird's` attention is an approximation of `BERT`'s full attention and is thus not **better** as `BERT's` full attention. It simply allows to apply transformer-based models to much longer sequences since BERT's quadratic memory requirement quickly becomes unbearable. Simply put, if we would have \\(\infty\\) compute & \\(\infty\\) time, BERT's attention is always better than block sparse attention (which we are going to discuss in this post). So, one should aim for attending to as many tokens as possible at a reasonable compute & memory cost.

If you wonder why we need more compute when working with longer sequences, (no worries!) just continue reading this post.

---

Some of the main question when working with normal `BERT`-like attention, one might have are:

* Do all the tokens really have to attend to all other tokens?
* Why not compute attention over only those tokens that are important?
* How to decide what tokens are important?
* How to attend only few tokens very efficiently?

---

In this blog post, we will try to answer those questions.

### What tokens to attend?

Unlike `BERT`-like attention, where every token attend every other token; we will try to build an empty `set` and fill it up with the tokens which any token should attend based on some ideas & intuition.

```python
>>> # let's consider following sentence as an example
>>> example = ['BigBird', 'is', 'now', 'available', 'in', 'HuggingFace', 'for', 'extractive', 'question', 'answering']

>>> # further let's assume, we're trying to understand the representation of 'available' i.e. 
>>> query_token = 'available'

>>> # We will initialize an empty `set` and fill up the tokens of our interest as we proceed in this section.
>>> key_tokens = set() # => currently 'available' token doesn't have anything to attend
```

Nearby tokens should be important because in a sentence (sequence of words), the current word is highly dependent on neighboring past & future tokens. This intuition is the idea behind the concept of `sliding attention`.

```python
>>> # considering `window_size = 3`, we will consider 1 token to left & 1 to right of 'available'
>>> # left token: 'now' ; right token: 'in'
>>> sliding_tokens = ["now", "available", "in"]

>>> # let's update our collection with the above tokens
>>> key_tokens.update(sliding_tokens)
```

**Long range dependencies:** For some tasks, it is crucial that the model is able to capture long range relationships between tokens. *E.g.*, in `question-answering` the model needs to compare each token of the context to the whole question to be able to figure out which part of the context is useful for a correct answer. If most of the context tokens would just attend to other context tokens, but not to the question, it becomes much harder for the model to filter important context tokens from less important context tokens.

Now, `BigBird` proposes two ways of allowing long-term attention dependencies while staying computationally efficient.

* **Global tokens:** Introduce some tokens which will attend to every token and which are attented by every token. Eg: *"HuggingFace is building nice libraries for easy NLP"*. Now, let's say *'building'* is defined as a global token, and model needs to know relation among *'NLP'* & *'HuggingFace'* for some task (Note: these 2 tokens are at two extremes); Now having *'building'* attend globally to all other tokens will probably help the model to assiciate *'NLP'* with *'HuggingFace'*.

```python
>>> # let's assume 1st & last token to be `global`, then
>>> global_tokens = ["BigBird", "answering"]

>>> # fill up global tokens in our key tokens collection
>>> key_tokens.update(global_tokens)
```

* **Random tokens:** Select some tokens randomly which will transfer information by transfering to other tokens which in turn can transfer to other tokens. This may reduce the cost of information travel from one token to other.

```python
>>> # now we can choose `r` token randomly from our example sentence
>>> # let's choose 'is' assuming `r=1`
>>> random_tokens = ["is"] # Note: it is chosen compleletly randomly; so it can be anything else also.

>>> # fill random tokens to our collection
>>> key_tokens.update(random_tokens)

>>> # it's time to see what all tokens get's into our `key_tokens`
>>> key_tokens
{'now', 'is', 'in', 'answering', 'available', 'BigBird'}

# Now, 'available' (query we choose in our 1st step) will attend only these tokens instead of attending the complete sequence
```

This way, we just need our query token to attend a subsequence & possibly model will be able to represent all the tokens nicely. The same approach will be used for all other tokens.
But remember, whole point here is to approximate `BERT`'s full attention as efficiently as possible. This is when `BigBird`'s block sparse attention comes into the picture.

### Understanding need of global, sliding, random keys with Graphs

Let's try to understand the need of `global`, `sliding` & `random` attention using graphs.

<img src="assets/global.png" width=250 height=250> </img>
<img src="assets/sliding.png" width=250 height=250> </img>
<img src="assets/random.png" width=250 height=250> </img> <br>

*The above figure shows `global` (left), `sliding` (middle) & `random` (right) connections respectively as a graph. Each node corresponds to a token and each line represents attention score. If no connection is made between 2 tokens, then attention score is assumed to 0.*

![](assets/graph.gif)
<img src="assets/full.png" width=230 height=230> </img>

***BigBird block sparse attention** is simply combination of sliding, global & random connections (total 15 connections) as shown in `gif` in left. While, graph of **normal attention** (right) will have all 21 connections (note: total 6 nodes are present). You can simply think of normal attention as all the tokens attending globally.*

**Normal attention:** Model can transfer information from one token to another token directly in a single layer, since each token is queried over every other token and is attended by every other tokens. Let's consider an example similar to what shown in above figures. If model needs to associate *'going'* with *'now'*, it can simply do that in a single layer since there is a direct connection joining both the tokens.

**Block sparse attention:** If model needs to share information between two nodes (or tokens), information will have to travel across various other nodes in the path for some of the tokens; since all the nodes are not directly connected in a single layer.
*Eg.* In case, model needs to associate *'going'* with *'now'*, if only sliding attention is present then for flow of information among those 2 tokens, will happen via this path: `going -> am -> i -> now` (i.e. it will have to travel over 2 other tokens). Hence, we may need multiple layers to capture the entire information of the sequence; which normal attention can capture in a single layer. This can amount to a time complexity of $O(n^2)$ because now model may need as many layers as sequence length. But if we introduce some global tokens, then information can travel via path: `going -> i -> now` (which is shorter); while if we introduce random connections it can travel via: `going -> am -> now`. This way with the help of random connections & global connections, information can travel very fastly (possibly in very few layers) via multiple short paths.

Think of the case when we have many global tokens, then we may not even need random tokens since there will be multiple short paths (involving global tokens) through which information can travel. This is the idea behind keeping `num_random_tokens = 0` when working with ETC (more on this in later sections).

*Note: We are assuming that attention matrix to be symmetric **i.e.** \\(\mathbf{A}_{ij} = \mathbf{A}_{ji}\\) since in a graph if some token **A** attends **B**, then **B** will also attend **A**. You can clearly see from attention matrix (figure shown in next section) that this assumption holds for most of the tokens.*

| Attention Type  | `global_tokens`   | `sliding_tokens` | `random_tokens`                    |
|-----------------|-------------------|------------------|------------------------------------|
| `original_full` | `n`               | 0                | 0                                  |
| `block_sparse`  | 2 x `block_size`  | 3 x `block_size` | `num_random_blocks` x `block_size` |

*`original_full` represents `BERT`'s attention while `block_sparse` represents `BigBird`'s attention. Wondering what the `block_size` is? We will cover that in later sections. For now consider it to be 1 for simplicity*

## BigBird block sparse attention

BigBird block sparse attention is just an efficient implementation of what we discussed above. Each token is attending some **global tokens**, **sliding tokens**, & **random tokens** instead of attending to **all** other tokens. The authors hardcoded the attention matrix for mutiple query components seperately; and used a cool trick to speed up training/inference process on gpu/tpu.

![BigBird block sparse attention](assets/attn.png)
*Note: on the top, we have 2 extra sentences. As you can notice, every token is just switched by one place in both sentence. This is how sliding attention is implemented. When `q[i]` is multiplied with `k[i,0:3]`, we will get a sliding attention score for `q[i]` (where `i` is index of element in sequence).*

You can find the actual implementation of `block_sparse` attention [here](https://github.com/vasudevgupta7/transformers/blob/5f2d6a0c93ca2017961199aa04a344b9b779d454/src/transformers/models/big_bird/modeling_big_bird.py#L513). This may look very scary 😨😨 now. But this article will surely ease your life in understanding the code.

### Global Attention

For global attention, each query is simply attending to all the other tokens in the sequence & is attended by every other token. Let's assume `Vasudev` (1st token) & `them` (last token) to be global (in above figure). You can see that these tokens are directly connected to all other tokens (blue boxes).

```python
# pseudo code

Q -> Query martix (seq_length, head_dim)
K -> Key matrix (seq_length, head_dim)

# 1st & last token attends all other tokens
Q[0] x [K[0], K[1], K[2], ......, K[n-1]]
Q[n-1] x [K[0], K[1], K[2], ......, K[n-1]]

# 1st & last token getting attended by all other tokens
K[0] x [Q[0], Q[1], Q[2], ......, Q[n-1]]
K[n-1] x [Q[0], Q[1], Q[2], ......, Q[n-1]]
```

### Sliding Attention

Key sequence is copied 2 times with each element shifted to right in one of the copy and to the left in the other copy. Now if we multiply query sequence vectors by these 3 sequences vectors, we will cover all the sliding tokens. Compute capacity of that
will be only `O(3xn)` or simply `O(n)`. Referring to the above picture, the orange boxes represents sliding attention. You can see 3 sequences in the top of figure with 2 of them shifted by one token (1 to the left, 1 to the right).

```python
# what we want to do
Q[i] x [K[i-1], K[i], K[i+1]] for i = 1:-1

# efficient implementation in code (assume dot product multiplication 👇)
[Q[0], Q[1], Q[2], ......, Q[n-2], Q[n-1]] x [K[1], K[2], K[3], ......, K[n-1], K[0]]
[Q[0], Q[1], Q[2], ......, Q[n-1]] x [K[n-1], K[0], K[1], ......, K[n-2]]
[Q[0], Q[1], Q[2], ......, Q[n-1]] x [K[0], K[1], K[2], ......, K[n-1]]

# Each sequence is getting mutiplied by only 3 sequences to keep `window_size = 3`.
# Some computations might be missing; this is just a rough idea.
```

### Random Attention

Random attention is ensuring that each query token will attend a few random tokens as well. For the actual implementation, this means that the model simply gathers some tokens randomly and computes their attention score.

```python
# r1, r2, r are some random indices; Note: r1, r2, r3 are different for each row 👇
Q[1] x [Q[r1], Q[r2], ......, Q[r]]
.
.
.
Q[n-2] x [Q[r1], Q[r2], ......, Q[r]]

# leaving 0th & (n-1)th token since they are already global
```

**Note:** The current implementation further divides sequence into blocks & each notation is defined w.r.to block instead of token. Let's discuss more in next section.

### Implementation

**Recap:** In regular BERT attention, a sequence of tokens **i.e. $X = x_1, x_2, ...., x_n$** are projected through a dense layer into **$Q,K,V$** and attention score (**$Z$**) is calculated as **$Z=Softmax(QK^T)$**. Now, in case of BigBird block sparse attention, we are going to similar operation only but with selective queries & keys.

Let's have a look at how bigbird block sparse attention is implemented in code. Assuming $b, r, s, g$ represents `block_size`, `num_random_blocks`, `num_sliding_blocks`, `num_global_blocks`, respectively. Visually, we can illustrate big bird block sparse attention with $b=4, r=1, g=2, s=3, d=5$ as computation among following:

<img src="assets/intro.png" width=500 height=250> </img>

Attention scores for \\({q}_{1}, {q}_{2}, {q}_{3:n-2}, {q}_{n-1}, {q}_{n}\\) are calculated seperately as described below:

---

Attention score for \\(\mathbf{q}_{1}\\) represented by $a_1$ where $a_1=Softmax(q_1 * K^T)$, is nothing but attention score between all the tokens in 1st block with all the other tokens in sequence.

![BigBird block sparse attention](assets/q1.png)
*$q_1$ represents 1st block, $g_i$ represents $i$ block. We are simply performing normal attention operation between $q_1$ & $g$ (i.e. all the keys).*

---

For calculating attention score for tokens in 2nd block, we are 1st gathering the 1st three blocks, last block, and fifth block. Then $a_2 = Softmax(q_2 * concat(k_1, k_2, k_3, k_5, k_7))$.

![BigBird block sparse attention](assets/q2.png)

*I am representing tokens by $g, r, s$ just to represent their nature explicitly (i.e. showing global, random, sliding tokens), else they are $k$ only.*

---

For calculating attention score for \\({q}_{3:n-2}\\), we will gather global, sliding, random keys & will simply calculate normal attention operation among all \\({q}_{3:n-2}\\) & gathered keys. Note that sliding keys are gathered using the special trick, we discussed in sliding attention section.

![BigBird block sparse attention](assets/q_middle.png)

---

For calculating attention score for tokens in last 2nd block (i.e. \\({q}_{n-1}\\)), we are 1st gathering the 1st blocks, last three block, and 3rd block. Then \\({a}_{n-1} = Softmax({q}_{n-1} * concat(k_1, k_3, k_5, k_6, k_7))\\). This is very similar to what we did for $q_2$.

![BigBird block sparse attention](assets/qlast_sec.png)

---

Attention score for \\(\mathbf{q}_{n}\\) represented by $a_n$ where $a_n=Softmax(q_n * K^T)$, is nothing but attention score between all the tokens in last block with all the other tokens in sequence. This is very similar to what we did for $q_1$.

![BigBird block sparse attention](assets/qlast.png)

---

Let's combine the above matrices to get the final attention matrix. This attention matrix can be used to get representation of all the tokens.

![BigBird block sparse attention](assets/block-sparse-attn.gif)

*`blue -> global blocks`, `red -> random blocks`, `orange -> sliding blocks` This attention matrix is just for illustration. During forward pass, we aren't storing `white` blocks & calculating weighted value matrix (i.e. representation of each token) directly from separated components discussed above.*

We have covered the most hardest part of block sparse attention i.e. its implementation. Now, you are good to go to read the code. Feel free to do that.

## Time & Memory complexity

| Attention Type  | Sequence length | Time & Memory Complexity |
|-----------------|-----------------|--------------------------|
| `original_full` | 512             | `T`                      |
|                 | 1024            | 4 x `T`                  |
|                 | 4096            | 64 x `T`                 |
| `block_sparse`  | 1024            | 2 x `T`                  |
|                 | 4096            | 8 x `T`                  |

*Comparison of time & space complexity of BERT attention and BigBird block sparse attention.*

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

<!-- 
## Benchmark

TODO
add benchmark of bigbird vs longformer or BERT (RTX 3090 24 GB)
-->

## ITC vs ETC

BigBird model can be trained using 2 different strategies: **ITC** & **ETC**. ITC (internal transformer construction) is simply what we discussed above. While in ETC (extended transformer construction), some extra tokens are made global such that they will attend / will be attented by all tokens.

ITC requires less compute since very few tokens are globals & model can still capture global information with them. On the other hand, ETC can be very helpful for the tasks in which we need lot of global tokens such as `question-answering` in which entire question should be global, with many tokens of context to be able to understand context; `summarization` since model needs to understand the overall context of very long paragraph to be able to summarize.

***Note:** In BigBird paper when working with ETC, number of random blocks are assumed to be 0 for training on many tasks. This favours our discussion in graph section.*

The table below summarizes ITC & ETC:

|                                              | ITC                                   | ETC                                  |
|----------------------------------------------|---------------------------------------|--------------------------------------|
| Attention Matrix with global attention       |<a href="https://www.codecogs.com/eqnedit.php?latex=A&space;=&space;\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A&space;=&space;\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" title="A = \begin{bmatrix} 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & & & & & & 1 \\ 1 & & & & & & 1 \\ 1 & & & & & & 1 \\ 1 & & & & & & 1 \\ 1 & & & & & & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{bmatrix}" /></a> | <a href="https://www.codecogs.com/eqnedit.php?latex=B&space;=&space;\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?B&space;=&space;\begin{bmatrix}&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;&&space;&&space;&&space;&&space;&&space;1&space;\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;\end{bmatrix}" title="B = \begin{bmatrix} 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & & & & & & 1 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{bmatrix}" /></a> |
| `global_tokens`   | 2 x `block_size`                      | `extra_tokens` + 2 x `block_size`     |
| `random_tokens`   | `num_random_blocks` x `block_size`    | `num_random_blocks` x `block_size`    |
| `sliding_tokens`  | 3 x `block_size`                      | 3 x `block_size`                      |

## Using BigBird with 🤗Transformers

You can use `BigBirdModel` just like any other 🤗 model. Let's see some code below:

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

There are total **3 checkpoints** available in **🤗Hub** (at the point of writing this article): [`bigbird-roberta-base`](https://huggingface.co/google/bigbird-roberta-base), [`bigbird-roberta-large`](https://huggingface.co/google/bigbird-roberta-large), [`bigbird-base-trivia-itc`](https://huggingface.co/google/bigbird-base-trivia-itc). The first two checkpoints come from pretraining `BigBirdForPretraining` with `masked_lm loss`; while the last one corresponds to the checkpoint after finetuning `BigBirdForQuestionAnswering` on `trivia-qa` dataset.

Let's have a look at minimal code you can write (in case you like to use your own PyTorch trainer), to use 🤗's BigBird model for fine-tuning on your tasks.

```python
# let's consider our task to be question-answering as an example

from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
import torch

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

# lets initialize bigbird model from pretrained weights with randomly initialized head on its top
model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-roberta-base", block_size=64, num_random_blocks=3)
tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
model.to(device)

dataset = "torch.utils.data.DataLoader object"
optimizer = "torch.optim object"
epochs = ...

# very minimal training loop
for e in range(epochs):
    for batch in dataset:
        model.train()
        batch = {k: batch[k].to(device) for k in batch}

        # forward pass
        output = model(**batch)

        # back-propogation
        output["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

# let's save final weights in local directory
model.save_pretrained("<YOUR-WEIGHTS-DIR>")

# let's push our weights to 🤗Hub
from huggingface_hub import ModelHubMixin
ModelHubMixin.push_to_hub("<YOUR-WEIGHTS-DIR>", model_id="<YOUR-FINETUNED-ID>")

# using finetuned model for inference
question = ["How are you doing?", "How is life going?"]
context = ["<some big context having ans-1>", "<some big context having ans-2>"]
batch = tokenizer(question, context, return_tensors="pt")
batch = {k: batch[k].to(device) for k in batch}

model = BigBirdForQuestionAnswering.from_pretrained("<YOUR-FINETUNED-ID>")
model.to(device)
with torch.no_grad():
    start_logits, end_logits = model(**batch).to_tuple()
    # now decode start_logits, end_logits with what ever strategy you want.

# Note:
# This was very minimal code (incase you want to use raw PyTorch) just for showing how BigBird can be used very easily
# I would suggest to use 🤗Trainer for lot of features
```

It's important to keep the following points in mind while working with big bird:

* Sequence length must be a multiple of block size i.e. `seqlen % block_size = 0`. You need not worry since 🤗Transformers will automatically `<pad>` (to smallest multiple of block size which is greater than sequence length) if batch sequence length is not a multiple of `block_size`.
* Currently, HuggingFace version **doesn't support ETC** and hence only 1st & last block will be global.
* Current implementation doesn't support `num_random_blocks = 0`.
* It's recommended by authors to set `attention_type = "original_full"` when sequence length < 1024.
* This must hold: `seq_length > global_token + random_tokens + sliding_tokens + buffer_tokens` where `global_tokens = 2 x block_size`, `sliding_tokens = 3 x block_size`, `random_tokens = num_random_blocks x block_size` & `buffer_tokens = num_random_blocks x block_size`. In case you fail to do that, 🤗Transformers will automatically switch `attention_type` to `original_full` with a warning.
* When using big bird as decoder (or using `BigBirdForCasualLM`), `attention_type` should be `original_full`. But you need not worry, 🤗Transformers will automatically switch `attention_type` to `original_full` in case you forget to do that.

## What's next?

[@patrickvonplaten](https://github.com/patrickvonplaten) has made a really cool [notebook](https://colab.research.google.com/drive/1DVOm1VHjW0eKCayFq1N2GpY6GR9M4tJP?usp=sharing) on how to evaluate `BigBirdForQuestionAnswering` on `trivia-qa` dataset. Feel free to play with big bird using that notebook.

You will soon find **BigBird Pegasus-like** model in the library and will be able to do easily perform **long document summarization**💥.

## End Notes

The original implementation of **block sparse attention matrix** can be found [here](https://github.com/google-research/bigbird/blob/master/bigbird/core/attention.py). You can find 🤗 version [here](https://github.com/huggingface/transformers/tree/master/src/transformers/models/big_bird).

**Feel free to raise an issue, incase you found something wrong here. Star 🌟 this repo if you found this helpful.**
