import torch
import numpy as np
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
from datasets import load_dataset

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
        start, end = model(input_ids=input_ids).to_tuple()
        _, start = start.max(dim=-1)
        _, end = end.max(dim=-1)
        start, end = start.item(), end.item()
    input_ids = input_ids[0].cpu().tolist()
    answer = tokenizer.decode(input_ids[start: end+1])
    orig_answer = tokenizer.decode(input_ids[example["start_token"]: example["end_token"]+1])

    if print_output:
        print(tokenizer.decode(input_ids[1:input_ids.index(tokenizer.sep_token_id)]))
        print("ORIGINAL:", orig_answer)
        print("MODEL:", answer, end="\n\n")

    example["orig_answer"] = orig_answer
    example["answer"] = answer
    example['match'] = 1 if answer == orig_answer else 0 # exact match
    return example

MODEL_ID = "important/ckpts-small-data/interrupted-natural-questions"

if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = BigBirdForQuestionAnswering.from_pretrained(MODEL_ID).to(device)
    tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-base-trivia-itc")

    val_dataset = load_dataset("json", data_files="data/nq-validation.jsonl")['train']
    np.random.seed(42)
    indices = np.random.randint(0, 18719, size=1000*2)
    val_dataset = val_dataset.select(indices)
    val_dataset = val_dataset.filter(lambda x: not (x['start_token'] == 0 and x['end_token'] == 0))

    val_dataset = val_dataset.map(get_answer)
    total = len(val_dataset)
    matched = len(val_dataset.filter(lambda x: x["match"] == 1))
    print("EM score:", (matched / total)*100, "%")
