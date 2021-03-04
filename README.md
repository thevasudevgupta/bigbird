# BigBird

BigBird is the transformer based model which is relying on `block sparse attention` instead of normal attention (as suggested in paper- Attention is all you need). It can handle sequence length `upto 4096` at a very low compute cost compared to BERT on that long sequences. It has achieved SOTA on various tasks involving very long sequences (typically > 1024) such as long documents summarization, question-answering with longer contexts.

## Big Bird block sparse attention

Paper suggested to do attention over `global tokens`, `sliding tokens`, & `random tokens` instead of doing it over complete sequence when sequence length is very large (>1024) as compute cost increase significantly in that case. Theoretically, compute complexity reduced to `n` from `n^2` this way. But practically, we need to use `gather` operation to combine all the keys (global + sliding + random) involved in block sparse attention matrix, which is very slow when using gpu/tpu.

Hence, Authors hardcoded attention matrix and used a simple (but cool) trick to speed up training/inference process on gpu/tpu and reduced the need for relying on `gather` operation.

### Global Attention

For global attention, each query is simply attending all the other tokens in sequence & is getting attended by every other token. `BigBird` implemented in `HuggingFace` is currently having 1st few tokens and last few tokens (depending on block size) as global tokens.

### Sliding Attention

Key sequence is copied 3 times with each element shifted to right in one of the copy & to the left in the other copy. Now if we multiply query sequence vectors by these 3 sequences vectors, we will cover all the sliding tokens. Compute capacity of that 
will be only O(3xn) or simpy O(n). Refer below figure for the clear idea:

### Random Attention

Random attention is ensuring that each query token will attend few random tokens as well. Here we will have to rely upon `gather` operation :( during implementation.

![ ](assets/block_sparse.png)

Current implementation further divides sequence into blocks and computation is performed over each block of tokens instead of over single token for making the whole process more efficient on gpu/tpu.

## ITC vs ETC

```md
ITC:
    global_tokens: 2 x block_size (1st & last block)
    random_tokens: r x block_size
    sliding_tokens: 3 x block_size

ETC:
    global_tokens: extra_tokens + 2 x block_size (1st & last block)
    random_tokens: r x block_size
    sliding_tokens: 3 x block_size
```

## End Notes

Original implementation of attention matrix can be found [here](https://github.com/google-research/bigbird/blob/master/bigbird/core/attention.py). I am currently adding `pytorch` version of BigBird to `HuggingFace`💥. I have commented a lot in `block_sparse_attention` method for everyone to be able to easily understand its implementation, you can refer [that](https://github.com/huggingface/transformers/pull/10183).

**Feel free to raise an issue, incase you found something wrong here. Star 🌟 this repo if you found this helpful.**
