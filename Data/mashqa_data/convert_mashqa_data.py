import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids

    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res


# Read the JSON data from file
with open('../../Data/train_data.json', 'r') as file:
	data = json.load(file)

converted_data = []

print("total len: ",len(data))

# Iterate through each item in the original data
count=0
for item in data:
    context = item['context']
    qas = []
    print(count)
    count=count+1
    # Iterate through each question-answer pair
    for qa in item['qas']:
        question = qa['question']
        answer = paraphrase(qa['answers'][0]['text'])
        
        # Create a new question-answer pair dictionary
        qa_dict = {
            'question': question,
            'answers': answer
        }
        
        qas.append(qa_dict)
    
    # Create a new dictionary for the context and question-answer pairs
    converted_item = {
        'context': context,
        'qas': qas
    }    
    converted_data.append(converted_item)
    if count%500==0:
        filename='mashqa_train_'+str(count)+'_data.json'
        print(filename)
        with open(filename, 'w') as outfile:
            json.dump(converted_data, outfile, indent=4)

# Write the converted data to a new JSON file
with open('mashqa_train_data.json', 'w') as outfile:
    json.dump(converted_data, outfile, indent=4)