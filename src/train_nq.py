
import torch
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

import wandb
import os

os.environ['WANDB_WATCH'] = "false"
os.environ['WANDB_PROJECT'] = "bigbird-natural-questions"
MODEL_ID = "google/bigbird-roberta-base"


def collate_fn(features, pad_id=0):
    def pad_elems(ls, pad_id, maxlen):
        while len(ls)<maxlen:
            ls.append(pad_id)
        return ls

    # dynamic padding
    maxlen = max([len(x['input_ids']) for x in features])
    input_ids = [pad_elems(x['input_ids'], pad_id, maxlen) for x in features]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "start_positions": torch.tensor([x['start_token'] for x in features], dtype=torch.long),
        "end_positions": torch.tensor([x['end_token'] for x in features], dtype=torch.long),
    }


if __name__ == "__main__":

    # "nq-training.jsonl" & "nq-validation.jsonl" are obtained from running `prepare_nq.py`
    tr_dataset = load_dataset("json", data_files="data/nq-training.jsonl")['train']
    val_dataset = load_dataset("json", data_files="data/nq-validation.jsonl")['train']

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
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_strategy="epoch",
        # logging_steps=4000,
        save_strategy="steps",
        run_name="bigbird-nq",
        disable_tqdm=False,
        load_best_model_at_end=True,
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

    trainer.train()
    trainer.save_model("best-model-natural-questions")
    wandb.finish()
