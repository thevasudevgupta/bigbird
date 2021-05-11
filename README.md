# BigBird

This repositary is tracking all my work related to porting [Google's BigBird](https://github.com/google-research/bigbird) to 🤗Transformers. I also trained 🤗's `BigBirdModel` (with suitable heads) on some of datasets mentioned in the paper: [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062). This repositary hosts those scripts as well!!

## Updates @ 🤗

| Description                                   | Dated          | Link                                                                |
|-----------------------------------------------|----------------|---------------------------------------------------------------------|
| 🤗's BigBird on TPUs                          | In progress    | [PR #11651](https://github.com/huggingface/transformers/pull/11651) |                                                  
| Ported `BigBird-Pegasus` @ **🤗Transformers** | May 7, 2021    | [PR #10991](https://github.com/huggingface/transformers/pull/10991) |
| Published blog post @ **🤗Blog**              | March 31, 2021 | [Link](https://huggingface.co/blog/big-bird)                        |
| Ported `BigBird-RoBERTa` @ **🤗Transformers** | March 30, 2021 | [PR #10183](https://github.com/huggingface/transformers/pull/10183) |

## Training BigBird

**Training on [`natural-questions`](https://huggingface.co/datasets/natural_questions) dataset**

```shell
# switch to natural-questions specific directory
cd natural-questions

# install requirements
pip3 install -r requirements.txt
```

For preparing the dataset for training, run the following commands:

```shell
# this will download ~ 100 GB dataset from 🤗Hub & prepare training data in `data/nq-training.jsonl`
PROCESS_TRAIN=True python3 prepare_nq.py

# for preparing validation data in `data/nq-validation.jsonl`
PROCESS_TRAIN=False python3 prepare_nq.py
```

Above commands will download dataset from 🤗Hub & will prepare it for training. Remember this will download ~100 GB of dataset, so you need to have good internet connection & enough space (~ 250 GB free space). Preparing dataset will take ~ 3 hours.

Now, for distributed training on several GPUs, run the following command:

```
# For distributed training (using nq-training.jsonl & nq-validation.jsonl) on multiple gpus
python3 -m torch.distributed.launch --nproc_per_node=2 train_nq.py
```

You can follow this [notebook](https://colab.research.google.com/github/vasudevgupta7/bigbird/blob/main/notebooks/evaluate_nq.ipynb) for evaluating the fine-tuned model.

| Checkpoint | [bigbird-roberta-natural-questions](https://huggingface.co/vasudevgupta/bigbird-roberta-natural-questions) |
|------------|------------------------------------------------------------------------------------------------------------|

To see how above checkpoint performs on QA task, checkout this: 

![](assets/infer-bigbird-nq.png)

*Context is just a tweet taken from 🤗 Twitter Handle.*💥💥💥
