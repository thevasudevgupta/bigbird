
import torch
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np

import wandb
import os

os.environ['WANDB_WATCH'] = "false"
os.environ['WANDB_PROJECT'] = "bigbird-natural-questions"
MODEL_ID = "google/bigbird-roberta-base"
SEED = 42
TRAIN_ON_SMALL = eval(os.environ.pop("TRAIN_ON_SMALL", "False"))


def collate_fn(features, pad_id=0):
    def pad_elems(ls, pad_id, maxlen):
        while len(ls)<maxlen:
            ls.append(pad_id)
        return ls

    # dynamic padding
    maxlen = max([len(x['input_ids']) for x in features])
    input_ids = [pad_elems(x['input_ids'], pad_id, maxlen) for x in features]
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    # padding mask
    attention_mask = input_ids.clone()
    attention_mask[attention_mask != pad_id] = 1
    attention_mask[attention_mask == pad_id] = 0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": torch.tensor([x['start_token'] for x in features], dtype=torch.long),
        "end_positions": torch.tensor([x['end_token'] for x in features], dtype=torch.long),
    }


if __name__ == "__main__":

    # "nq-training.jsonl" & "nq-validation.jsonl" are obtained from running `prepare_nq.py`
    tr_dataset = load_dataset("json", data_files="data/nq-training.jsonl")['train']
    val_dataset = load_dataset("json", data_files="data/nq-validation.jsonl")['train']

    if TRAIN_ON_SMALL:
        np.random.seed(SEED)
        indices = np.random.randint(0, 298152, size=8000)
        tr_dataset = tr_dataset.select(indices)
        np.random.seed(SEED)
        indices = np.random.randint(0, 18719, size=1000)
        val_dataset = val_dataset.select(indices)

    print(tr_dataset, val_dataset)

    tokenizer = BigBirdTokenizer.from_pretrained(MODEL_ID)
    model = BigBirdForQuestionAnswering.from_pretrained(MODEL_ID, block_size=64, num_random_blocks=3, attention_type="block_sparse")

    args = TrainingArguments(
        output_dir="bigbird-nq-output-dir",
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        # eval_steps=4000,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        group_by_length=True,
        learning_rate=7e-5,
        num_train_epochs=3,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        run_name="bigbird-nq",
        disable_tqdm=False,
        # load_best_model_at_end=True,
        report_to="wandb",
        remove_unused_columns=False,
        fp16=True,
    )
    print("Batch Size", args.train_batch_size)
    print("Parallel Mode", args.parallel_mode)

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        train_dataset=tr_dataset,
        eval_dataset=val_dataset,
    )
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.save_model("interrupted-natural-questions")
    wandb.finish()
