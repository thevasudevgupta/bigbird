# BigBird

## Updates @ 🤗

| Description                                   | Dated          | Link                                                                |
|-----------------------------------------------|----------------|---------------------------------------------------------------------|
| 🤗's BigBird on TPUs                          | In progress    | -                                                                   |
| Working on porting `BigBird-Pegasus`          | May 7, 2021    | [PR #10991](https://github.com/huggingface/transformers/pull/10991) |
| Published blog post @ **🤗Blog**              | March 31, 2021 | [Link](https://huggingface.co/blog/big-bird)                        |
| Ported `BigBird-RoBERTa` @ **🤗Transformers** | March 30, 2021 | [PR #10183](https://github.com/huggingface/transformers/pull/10183) |

## Running Scripts from this repositary

I am training `BigBirdModel` (with suitable heads) on several datasets mentioned in paper. To reproduce my results, follow below code:

**Training on [`natural-questions`](https://huggingface.co/datasets/natural_questions) dataset**

```shell
# switch to natural-questions specific directory
cd natural-questions

# install requirements
pip3 install -r requirements.txt

# this will download ~ 100 GB dataset from 🤗Hub & prepare training data in `data/nq-training.jsonl`
PROCESS_TRAIN=True python3 prepare_nq.py

# for preparing validation data in `data/nq-validation.jsonl`
PROCESS_TRAIN=False python3 prepare_nq.py

# For distributed training (using nq-training.jsonl & nq-validation.jsonl) on multiple gpus
python3 -m torch.distributed.launch --nproc_per_node=2 train_nq.py

# for evaluation on validation data
python3 evaluate_nq.py
```
