import numpy as np
import torch
from datasets import load_dataset

from params import CATEGORY_MAPPING
from train_nq import BigBirdForNaturalQuestions
from transformers import BigBirdTokenizer

CATEGORY_MAPPING = {v: k for k, v in CATEGORY_MAPPING.items()}

# def get_answer(question, context):
#     inputs = tokenizer(question, context, max_length=4096, padding=True, truncate=True, return_tensors="pt")
#     inputs = {k: inputs[k].to(device) for k in inputs}
#     input_ids = inputs["input_ids"][0]
#     with torch.no_grad():
#         start, end = model(**inputs).to_tuple()
#         _, start = start.max(dim=-1)
#         _, end = end.max(dim=-1)
#         start, end = start.item(), end.item()
#     answer = tokenizer.decode(input_ids[start: end+1])
#     return answer


def get_answer(example, print_output=True):
    input_ids = example["input_ids"]
    input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
    input_ids.unsqueeze_(0)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        _, start = outputs["start_logits"].max(dim=-1)
        _, end = outputs["end_logits"].max(dim=-1)
        start, end = start.item(), end.item()
        _, category = outputs["cls_out"].max(dim=-1)

    predicted_category = CATEGORY_MAPPING[category.item()]
    original_category = CATEGORY_MAPPING[example["category"]]

    input_ids = input_ids[0].cpu().tolist()
    answer = tokenizer.decode(input_ids[start : end + 1])
    orig_answer = tokenizer.decode(
        input_ids[example["start_token"] : example["end_token"] + 1]
    )

    if print_output:
        print(tokenizer.decode(input_ids[1 : input_ids.index(tokenizer.sep_token_id)]))
        print("original_category:", original_category)
        print("predicted_category:", predicted_category)
        if original_category in ["long", "short"]:
            print("original_answer:", orig_answer)
            print("predicted_answer:", answer, end="\n\n")

    example["orig_answer"] = orig_answer
    example["answer"] = answer
    if predicted_category in ["yes", "no", "null"]:
        example["match"] = 1 if predicted_category == original_category else 0
    else:
        example["match"] = 1 if answer == orig_answer else 0  # exact match
    return example


MODEL_ID = "important/small-good-data/checkpoint-750"

if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = BigBirdForNaturalQuestions.from_pretrained(MODEL_ID).to(device)
    tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

    val_dataset = load_dataset("json", data_files="data/nq-validation.jsonl")["train"]
    val_dataset = val_dataset.filter(lambda x: x["category"] != 0)
    np.random.seed(42)
    indices = np.random.randint(0, 9000, size=1000)
    val_dataset = val_dataset.select(indices)

    val_dataset = val_dataset.map(get_answer)
    total = len(val_dataset)
    matched = len(val_dataset.filter(lambda x: x["match"] == 1))
    print("EM score:", (matched / total) * 100, "%")
