# TRAIN_ON_SMALL=True python3 -m torch.distributed.launch --nproc_per_node=2 train_nq.py

import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset

from params import (
    FP16,
    GROUP_BY_LENGTH,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_ID,
    SCHEDULER,
    SEED,
    WARMUP_STEPS,
)
from transformers import (
    BigBirdForQuestionAnswering,
    BigBirdTokenizer,
    Trainer,
    TrainingArguments,
)

os.environ["WANDB_WATCH"] = "false"
os.environ["WANDB_PROJECT"] = "bigbird-natural-questions"
TRAIN_ON_SMALL = eval(os.environ.pop("TRAIN_ON_SMALL", "False"))


RESUME_TRAINING = None


def collate_fn(features, pad_id=0, threshold=1024):
    def pad_elems(ls, pad_id, maxlen):
        while len(ls) < maxlen:
            ls.append(pad_id)
        return ls

    maxlen = max([len(x["input_ids"]) for x in features])
    # avoid attention_type switching
    if maxlen < threshold:
        maxlen = threshold

    # dynamic padding
    input_ids = [pad_elems(x["input_ids"], pad_id, maxlen) for x in features]
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    # padding mask
    attention_mask = input_ids.clone()
    attention_mask[attention_mask != pad_id] = 1
    attention_mask[attention_mask == pad_id] = 0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": torch.tensor(
            [x["start_token"] for x in features], dtype=torch.long
        ),
        "end_positions": torch.tensor(
            [x["end_token"] for x in features], dtype=torch.long
        ),
        "pooler_label": torch.tensor([x["category"] for x in features]),
    }


class BigBirdForNaturalQuestions(BigBirdForQuestionAnswering):
    """BigBirdForQuestionAnswering with CLS Head over the top for predicting category"""

    def __init__(self, config):
        super().__init__(config, add_pooling_layer=True)
        self.cls = nn.Linear(config.hidden_size, 5)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        pooler_label=None,
    ):

        outputs = super().forward(input_ids, attention_mask=attention_mask)
        cls_out = self.cls(outputs.pooler_output)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            start_loss = loss_fct(outputs.start_logits, start_positions)
            end_loss = loss_fct(outputs.end_logits, end_positions)

            if pooler_label is not None:
                cls_loss = loss_fct(cls_out, pooler_label)
                loss = (start_loss + end_loss + cls_loss) / 3
            else:
                loss = (start_loss + end_loss) / 2

        return {
            "loss": loss,
            "start_logits": outputs.start_logits,
            "end_logits": outputs.end_logits,
            "cls_out": cls_out,
        }


if __name__ == "__main__":

    # #
    # model = BigBirdForNaturalQuestions.from_pretrained(MODEL_ID).cuda()
    # s = torch.ones(2,).long().cuda()
    # o = model(torch.ones(2, 512).long().cuda(), pooler_label=s, start_positions=s, end_positions=s)
    # print(o)
    # exit()
    # #

    # "nq-training.jsonl" & "nq-validation.jsonl" are obtained from running `prepare_nq.py`
    tr_dataset = load_dataset("json", data_files="data/nq-training.jsonl")["train"]
    val_dataset = load_dataset("json", data_files="data/nq-validation.jsonl")["train"]

    if TRAIN_ON_SMALL:
        # this will run for ~1 day
        np.random.seed(SEED)
        indices = np.random.randint(0, 298152, size=8000)
        tr_dataset = tr_dataset.select(indices)
        np.random.seed(SEED)
        indices = np.random.randint(0, 9000, size=1000)
        val_dataset = val_dataset.select(indices)

    # tr_dataset = tr_dataset.filter(lambda x: x['category'] != 0)
    # val_dataset = val_dataset.filter(lambda x: x['category'] != 0)
    print(tr_dataset, val_dataset)

    # for data in torch.utils.data.DataLoader(tr_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False):
    #     del data['input_ids']
    #     print(data)
    #     break
    # exit()

    tokenizer = BigBirdTokenizer.from_pretrained(MODEL_ID)
    model = BigBirdForNaturalQuestions.from_pretrained(
        MODEL_ID, gradient_checkpointing=True
    )

    args = TrainingArguments(
        output_dir="bigbird-nq-complete-tuning",
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        # eval_steps=4000,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        group_by_length=GROUP_BY_LENGTH,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=SCHEDULER,
        num_train_epochs=MAX_EPOCHS,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        run_name="bigbird-nq-complete-tuning",
        disable_tqdm=False,
        # load_best_model_at_end=True,
        report_to="wandb",
        remove_unused_columns=False,
        fp16=FP16,
        label_names=[
            "pooler_label",
            "start_positions",
            "end_positions",
        ],  # it's important to log eval_loss
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
