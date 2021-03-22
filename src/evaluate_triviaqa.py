# this code is adapted from notebook created by @patrick
import os
import datasets
import torch
from transformers import BigBirdTokenizer, BigBirdForQuestionAnswering

DATA_DIR = "data"

# define the mapping function
def format_dataset(example):
    # the context might be comprised of multiple contexts => me merge them here
    example["context"] = " ".join(("\n".join(example["entity_pages"]["wiki_context"])).split("\n"))
    example["targets"] = example["answer"]["aliases"]
    example["norm_target"] = example["answer"]["normalized_value"]
    return example

def evaluate(example):
    def get_answer(question, context):
        # encode question and context so that they are seperated by a tokenizer.sep_token and cut at max_length
        encoding = tokenizer(question, context, return_tensors="pt", max_length=4096, padding="max_length", truncation=True)
        input_ids = encoding.input_ids.to(device)
        attention_mask = encoding.attention_mask.to(device)

        # The scores for the possible start token and end token of the answer are retrived
        # wrap the function in torch.no_grad() to save memory
        with torch.no_grad():
            start_scores, end_scores = model(input_ids=input_ids, attention_mask=attention_mask).to_tuple()

        # Let's take the most likely token using `argmax` and retrieve the answer
        all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())

        answer_tokens = all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1]
        answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens)).replace('"', '')  # remove space prepending space token and remove unnecessary '"'
        
        return answer

    # save the model's output here
    example["output"] = get_answer(example["question"], example["context"])

    # save if it's a match or not
    example["match"] = (example["output"] in example["targets"]) or (example["output"] == example["norm_target"])

    return example


if __name__ == '__main__':

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    validation_dataset = datasets.load_dataset("trivia_qa", "rc", split="validation")
    print(validation_dataset)

    # map the dataset and throw out all unnecessary columns
    validation_dataset = validation_dataset.map(format_dataset, remove_columns=["search_results", "question_source", "entity_pages", "answer", "question_id"])
    validation_dataset = validation_dataset.filter(lambda x: len(x["context"]) > 0)

    validation_dataset = validation_dataset.filter(lambda x: (len(x['question']) + len(x['context'])) < 4 * 4096)

    tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-base-trivia-itc")
    model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-base-trivia-itc").to(device)

    validation_dataset = validation_dataset.map(evaluate)
    validation_dataset.to_csv(os.path.join(DATA_DIR, "predictions.csv"))

    print(f"\nNum Correct examples: {sum(validation_dataset['match'])}/{len(validation_dataset)}")
    wrong_results = validation_dataset.filter(lambda x: x['match'] is False)
    
    print(f"\nWrong examples: ")
    wrong_results.map(lambda x, i: print(f"{i} - Output: {x['output']} - Target: {x['norm_target']}"), with_indices=True)
