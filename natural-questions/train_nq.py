
import torch
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np

import wandb
import os

os.environ['WANDB_WATCH'] = "false"
os.environ['WANDB_PROJECT'] = "bigbird-natural-questions"
TRAIN_ON_SMALL = eval(os.environ.pop("TRAIN_ON_SMALL", "False"))

from params import (
    MODEL_ID,
    SEED,
    GROUP_BY_LENGTH,
    LEARNING_RATE,
    MAX_EPOCHS,
    FP16,
)

RESUME_TRAINING = None

def collate_fn(features, pad_id=0, threshold=1024):
    def pad_elems(ls, pad_id, maxlen):
        while len(ls)<maxlen:
            ls.append(pad_id)
        return ls

    maxlen = max([len(x['input_ids']) for x in features])
    # avoid attention_type switching
    if maxlen < threshold:
        maxlen = threshold

    # dynamic padding
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
        "pooler_label": torch.tensor([x["category"] for x in features]),
    }


if __name__ == "__main__":

    # "nq-training.jsonl" & "nq-validation.jsonl" are obtained from running `prepare_nq.py`
    tr_dataset = load_dataset("json", data_files="data/nq-training.jsonl")['train']
    val_dataset = load_dataset("json", data_files="data/nq-validation.jsonl")['train']

    if TRAIN_ON_SMALL:
        # this will run for ~1 day
        np.random.seed(SEED)
        indices = np.random.randint(0, 298152, size=8000*2)
        tr_dataset = tr_dataset.select(indices)
        np.random.seed(SEED)
        indices = np.random.randint(0, 18719, size=1000*2)
        val_dataset = val_dataset.select(indices)

    # let's try on samples without `cls`
    # tr_dataset = tr_dataset.filter(lambda x: not (x['start_token'] == 0 and x['end_token'] == 0))
    # val_dataset = val_dataset.filter(lambda x: not (x['start_token'] == 0 and x['end_token'] == 0))
    print(tr_dataset, val_dataset)

    tokenizer = BigBirdTokenizer.from_pretrained(MODEL_ID)

    # for data in torch.utils.data.DataLoader(tr_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False):
    #     pass
    # exit()

    # TODO: add pooler label
    model = BigBirdForQuestionAnswering.from_pretrained(MODEL_ID, gradient_checkpointing=True)

    args = TrainingArguments(
        output_dir="bigbird-nq-output-dir",
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        # eval_steps=4000,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        group_by_length=GROUP_BY_LENGTH,
        learning_rate=LEARNING_RATE,
        num_train_epochs=MAX_EPOCHS,
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="steps",
        save_steps=300,
        run_name="bigbird-nq",
        disable_tqdm=False,
        # load_best_model_at_end=True,
        report_to="wandb",
        remove_unused_columns=False,
        fp16=FP16,
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
        trainer.train(resume_from_checkpoint=RESUME_TRAINING)
        trainer.save_model("final-model")
    except KeyboardInterrupt:
        trainer.save_model("interrupted-natural-questions")
    wandb.finish()
