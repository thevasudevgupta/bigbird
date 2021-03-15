# BigBird

`BigBird` is the transformer based model which is relying on `block sparse attention` instead of normal attention (which can be found in `BERT`). It can handle sequence length `upto 4096` at a very low compute cost compared to `BERT` on that long sequences. It has achieved SOTA on various tasks involving very long sequences (typically > 1024) such as long documents summarization, question-answering with longer contexts.

## Big Bird block sparse attention

Paper suggested to do attention over `global tokens`, `sliding tokens`, & `random tokens` instead of doing it over complete sequence when sequence length is very large (>1024) as compute cost increase significantly in that case. Theoretically, compute complexity reduced to `n` from `n^2` this way. But practically, we need to use `gather` operation to combine all the keys (global + sliding + random) involved in block sparse attention matrix, which is very slow when using gpu/tpu.

Hence, Authors hardcoded attention matrix and used a simple (but cool) trick to speed up training/inference process on gpu/tpu and reduced the need for relying on `gather` operation. Let's figure out more on that ...

### Global Attention

For global attention, each query is simply attending all the other tokens in sequence & is getting attended by every other token. `BigBird` implemented in `HuggingFace` is currently having 1st few tokens and last few tokens (depending on block size) as global tokens.

### Sliding Attention

Key sequence is copied 3 times with each element shifted to right in one of the copy & to the left in the other copy. Now if we multiply query sequence vectors by these 3 sequences vectors, we will cover all the sliding tokens. Compute capacity of that
will be only `O(3xn)` or simpy `O(n)`. Refer below figure for the clear idea. You can clearly see 3 sequences in the top of figure with 2 of them switched by one token.

### Random Attention

Random attention is ensuring that each query token will attend few random tokens as well. Here we will have to rely upon `gather` operation :( during implementation.

![ ](assets/block_sparse.png)

Current implementation further divides sequence into blocks and computation is performed over each block of tokens instead of over single token for making the whole process more efficient on gpu/tpu.

## ITC vs ETC

Further, BigBird model is pretrained using 2 different strategies: `ITC` & `ETC`. `ITC` is simply what we discussed above. While in `ETC`, some more tokens are made global such that they will attend / will get attented by all tokens. Paper claimed that this can lead to increase in performance on several tasks.

```md
ITC:
    global_tokens: 2 x block_size (1st & last block)
    random_tokens: num_random_blocks x block_size
    sliding_tokens: 3 x block_size

ETC:
    global_tokens: extra_tokens + 2 x block_size (1st & last block)
    random_tokens: num_random_blocks x block_size
    sliding_tokens: 3 x block_size
```

## Working with Hugging Face transformers

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

There are total 3 checkpoints available in huggingface_hub (at the point of writing this article): `bigbird-roberta-base`, `bigbird-roberta-large`, `bigbird-base-trivia-itc`. First 2 checkpoints are the checkpoints made available after pretrained `BigBirdForPretraining` while the last one corresponds to the checkpoint after finetuning `BigBirdForQuestionAnswering` on `trivia-qa` dataset.

## End Notes

Original implementation of block sparse attention matrix can be found [here](https://github.com/google-research/bigbird/blob/master/bigbird/core/attention.py). You can find `HuggingFace` version [here](https://github.com/huggingface/transformers/pull/10183).

**Feel free to raise an issue, incase you found something wrong here. Star 🌟 this repo if you found this helpful.**
