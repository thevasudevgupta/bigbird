
# ['document']["tokens"]['token'] & ['document']["tokens"]['is_html']
# ['document']["html"] -> extreme uncleaned
# ["question"]["text"] or ["question"]["tokens"]
# ['annotations] -> 5 dict
# short ans VS long ans

from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

PRETRAINED_MODEL_ID = "google/bigbird-roberta-base"
SPLIT = ["train[:1%]", "validation[:1%]"]

def collate_fn(features):

    return

if __name__ == "__main__":

    data = load_dataset("natural_questions", split=SPLIT)
    tr_data, val_data = data
    print(data)

    model = BigBirdForQuestionAnswering.from_pretrained(PRETRAINED_MODEL_ID)
    tokenizer = BigBirdTokenizer.from_pretrained(PRETRAINED_MODEL_ID)

    args = TrainingArguments(
        output_dir="bigbird-nq",
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
        run_name="bigbird-nq-experiment1",
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
        train_dataset=tr_data,
        eval_dataset=val_data,
        )
