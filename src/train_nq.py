import os
os.environ['WANDB_WATCH'] = "false"
os.environ['WANDB_PROJECT'] = "bigbird-natural-questions"

MODEL_ID = "bigbird-roberta-base"

from transformers import BigBirdModel, BigBirdTokenizer
from transformers import TrainingArguments, Trainer
import wandb


def collate_fn(features):

  context = [x["document"]["summary"]["text"] for x in features]
  question = [x["question"]["text"] for x in features]
  answer = [x["answers"][0]["text"] for x in features]

  # should not eliminate special tokens since question and context are should have `SEP` in middle
  inputs = tokenizer(question, context, return_tensors="pt", padding="max_length", truncation=True, max_length=SRC_MAXLEN)
  labels = tokenizer(answer, return_tensors="pt", padding=True, truncation=True, max_length=TGT_MAXLEN)

  return {
      "input_ids": inputs.input_ids,
      "attention_mask": inputs.attention_mask,
      "decoder_input_ids": labels.input_ids,
      "labels": labels.input_ids,
      "decoder_attention_mask": labels.attention_mask,
  }


if __name__ == "__main__":

    tr_dataset, val_dataset = 

    model = BigBirdModel.from_pretrained(MODEL_ID, block_size=64, num_random_blocks=3, attention_type="block_sparse")
    tokenizer = BigBirdTokenizer.from_pretrained(MODEL_ID)

    args = TrainingArguments(
        output_dir="bigbird2bigbird-narrative-qa",
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        # eval_steps=4000,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=10,
        logging_strategy="steps",
        logging_steps=4000,
        save_strategy="epoch",
        run_name="bigbird2bigbird-narrative-qa-experiment1",
        disable_tqdm=False,
        load_best_model_at_end=True,
        report_to="wandb",
        remove_unused_columns=False,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        train_dataset=tr_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model("final_model-nq")
    wandb.finish()