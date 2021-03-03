# BigBird

BigBird is the transformer based model which is relying on `block sparse attention` instead of normal attention (suggested in paper- Attention is all you need). It can handle sequence length `upto 4096` at a very low compute cost compared to BERT on that much longer sequences. It has achieved SOTA on various tasks involving very long sequences (typically > 1024) such as summarization, question-answering with longer contexts.

## Big Bird block sparse attention

Paper suggested to do attention over `global tokens`, `sliding tokens`, & `random tokens` instead of doing it over complete sequence when sequence length is very large (>1024) as compute cost increase significantly in that case. Theoretically, compute complexity reduced to `n` from `n^2` this way. But practically, we need to use `gather` operation to combine all the keys involved in block sparse attention matrix, which is very slow when using gpu/tpu.

Hence, Authors hardcoded attention matrix and used a simple (but cool) trick to speed up training/inference process on gpu/tpu and reduce the need for relying on `gather` operation. Refer below figure for the clear idea:

![](assets/block_sparse.png)

Key sequence is copied 3 times with each element is shifted to right in one of the copy & to the left in the other copy.

Current implementation further divides sequence into blocks and computation is performed over each block of tokens instead of over single token for making the whole process more efficient on gpu/tpu.

## End Notes

Original implementation of attention matrix can be found [here](https://github.com/google-research/bigbird/blob/master/bigbird/core/attention.py). I am currently adding `pytorch` version of BigBird to HuggingFace. You can refer [this](https://github.com/huggingface/transformers/pull/10183) for the time being.
git