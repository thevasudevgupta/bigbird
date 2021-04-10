
# PUNCTUATION_SET_TO_EXCLUDE = ['‘', '’', '´', '`', "''", "^", "``"]

def _get_single_answer(example):
    def choose_first(answer, is_long_answer=False):
        assert isinstance(answer, list)
        if len(answer) == 1:
            answer = answer[0]
            return {k: [answer[k]] for k in answer} if is_long_answer else answer
        for a in answer:
            if is_long_answer:
                a = {k: [a[k]] for k in a}
            if len(a['start_token']) > 0:
                break
        return a

    answer = {
        "id": example['id']
    }
    annotation = example['annotations']
    yes_no_answer = annotation['yes_no_answer']
    if 0 in yes_no_answer or 1 in yes_no_answer:
        answer['category'] = ['yes'] if 1 in yes_no_answer else ['no']
        answer['start_token'] = answer['end_token'] = []
        answer['start_byte'] = answer['end_byte'] = []
        answer['text'] = ['<cls>']
    else:
        answer['category'] = ['short']
        out = choose_first(annotation['short_answers'])
        if len(out['start_token']) == 0:
            # answer will be long if short is not available
            answer['category'] = ['long']
            out = choose_first(annotation['long_answer'], is_long_answer=True)
            out['text'] = []
        answer.update(out)

    # disregard some samples
    if len(answer['start_token']) > 1 or answer['start_token'] == answer['end_token']:
        answer['remove_it'] = True
    else:
        answer['remove_it'] = False

    cols = ['start_token', 'end_token', 'start_byte', 'end_byte', 'text']
    if not all([isinstance(answer[k], list) for k in cols]):
        raise ValueError("Issue in ID", example['id'])

    # print(answer)
    return answer


def get_context_and_ans(example, assertion=False):
    """ Gives new context after removing <html> & new answer tokens as per new context """
    answer = _get_single_answer(example)
    # bytes are of no use
    del answer['start_byte']
    del answer['end_byte']

    if len(answer['start_token']) == 0:
        return {
            "context": "None",
            "answer": {
                'start_token': -1,
                'end_token': -1,
                'span': "None",
            }
        }

    cols = ['start_token', 'end_token']
    answer.update({k: answer[k][0] if len(answer[k])>0 else answer[k] for k in cols})

    doc = example['document']['tokens']
    start_token = answer['start_token']
    end_token = answer['end_token']

    context = []
    for i in range(len(doc['token'])):
        if not doc['is_html'][i]:
            context.append(doc['token'][i])
        else:
            if answer['start_token'] > i:
                start_token -= 1
            if answer['end_token'] > i:
                end_token -= 1
    new = " ".join(context[start_token: end_token])

    # checking above code
    if assertion:
        """ checking if above code is working as expected for all the samples """
        is_html = doc["is_html"][answer['start_token']: answer['end_token']]
        old = doc['token'][answer['start_token']: answer['end_token']]
        old = ' '.join([old[i] for i in range(len(old)) if not is_html[i]])
        if new != old:
            print("ID:", example['id'])
            print('New:', new, end='\n')
            print('Old:', old, end='\n\n')

    # print(start_token, end_token - 1, new)
    # context = [c.strip()  ]
    # out = []
    # for c in context:
    #     if c not in PUNCTUATION_SET_TO_EXCLUDE:
    #         out.append(c.strip())
    #     else:
    #         answer['start_token'] -= 1
    #         answer['end_token'] -= 1
    # context = out

    return {
        'context': " ".join(context),
        'answer': {
            "start_token": start_token,
            "end_token": end_token - 1, # this makes inclusive
            "span": new,
        }
    }


def get_strided_contexts_and_ans(example, tokenizer, doc_stride=2048, max_length=4096, assertion=True):
    # overlap will be of doc_stride - q_len

    out = get_context_and_ans(example, assertion=assertion)
    answer = out['answer']
    if answer['start_token'] == -1:
        return {
        "example_id": example['id'],
        "input_ids": [[-1]],
        "answers_start_token": [-1],
        "answers_end_token": [-1],
    }

    input_ids = tokenizer(example['question']['text'], out['context']).input_ids

    splitted_context = out['context'].split()
    complete_end_token = splitted_context[answer['end_token']]
    answer['start_token'] = len(tokenizer(" ".join(splitted_context[:answer['start_token']]), add_special_tokens=False).input_ids)
    answer['end_token'] = len(tokenizer(" ".join(splitted_context[:answer['end_token']]), add_special_tokens=False).input_ids)

    q_len = input_ids.index(tokenizer.sep_token_id) + 1
    answer['start_token'] += q_len
    answer['end_token'] += q_len

    # fixing end token
    num_sub_tokens = len(tokenizer(complete_end_token, add_special_tokens=False).input_ids)
    if num_sub_tokens > 1:
        answer['end_token'] += num_sub_tokens - 1

    old = input_ids[answer['start_token']: answer['end_token']+1] # right & left are inclusive
    start_token = answer['start_token']
    end_token = answer['end_token']

    if assertion:
        """ This won't match exactly because of extra gaps => visaully inspect everything """
        new = tokenizer.decode(old)
        if answer["span"] != new:
            print("ISSUE IN TOKENIZATION")
            print("OLD:", answer["span"])
            print("NEW:", new, end="\n\n")

    if len(input_ids) <= max_length:
        return {
            "example_id": example['id'],
            "input_ids": [input_ids],
            "answers_start_token": [answer["start_token"]],
            "answers_end_token": [answer["end_token"]],
        }

    q_indices = input_ids[:q_len]
    doc_start_indices = range(q_len, len(input_ids), max_length-doc_stride)

    inputs = []
    answers_start_token = []
    answers_end_token = []
    for i in doc_start_indices:
        end_index = i + max_length - q_len
        slice = input_ids[i: end_index]
        inputs.append(q_indices + slice)
        assert len(inputs[-1]) <= max_length, "Issue in truncating length"
        if start_token >= i and end_token <= end_index-1:
            start_token = start_token - i + q_len
            end_token = end_token - i + q_len
        else:
            start_token = 0
            end_token = 0
        new = inputs[-1][start_token: end_token+1]
        answers_start_token.append(start_token)
        answers_end_token.append(end_token)
        if assertion:
            """ checking if above code is working as expected for all the samples """
            if new != old and new != [tokenizer.cls_token_id]:
                print("ISSUE in strided for ID:", example['id'])
                print('New:', tokenizer.decode(new))
                print('Old:', tokenizer.decode(old), end='\n\n')
        if slice[-1] == tokenizer.sep_token_id:
            break

    return {
        "example_id": example['id'],
        "input_ids": inputs,
        "answers_start_token": answers_start_token,
        "answers_end_token": answers_end_token,
    }


def prepare_inputs():
    return


if __name__ == "__main__":
    """ Testing area """
    from datasets import load_dataset
    from transformers import BigBirdTokenizer

    data = load_dataset('natural_questions')
    tok = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
    # sample with less seqlen = 3439

    # eg = data['train'][84593]
    # o = get_strided_contexts_and_ans(eg, tok)
    # print("QUESTION", eg['question']['text'])
    # for i in range(len(o['answers_start_token'])):
    #     a = o['answers_start_token'][i]
    #     b = o['answers_end_token'][i]
    #     print("ANSWER", tok.decode(o["input_ids"][i][a: b+1]), a, b)

    data = data['train'].map(get_strided_contexts_and_ans, fn_kwargs=dict(tokenizer=tok, assertion=True))
