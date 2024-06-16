from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

import pandas as pd
from datasets import load_dataset
import torch


#===============================================Load Model===============================================================
print("=========================================Load Model=========================================================")
model_checkpoint = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForCausalLM.from_pretrained('./gpt2-bioasq')

special_tokens_dict = {'additional_special_tokens': ['[EOC]','[SOC]','[SOR]', '[EOR]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token_id
model.resize_token_embeddings(len(tokenizer))
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(torch.device(device))
print("=========================================Load Model Done====================================================")

#===============================================Inference===============================================================
print("=========================================Inference===========================================================")
df_test = pd.read_csv('test_data.csv')
test = df_test['text'].to_list()
i=1
context = []
actual = []
response = []

for instance in test:
    text = instance.split('[EOC]')[0]+'[EOC]'
    gold = instance.split('[SOR]')[1].split('[EOR]')[0]
    try:
        input_ids = tokenizer.encode(text[-900:], return_tensors='pt').to(device)
    except:
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    context.append(text)
    actual.append(gold)
    sample_output = model.generate(input_ids, 
                                   do_sample=True, 
                                   max_length=input_ids.shape[1]+50, 
                                   top_k=0, 
                                   pad_token_id=tokenizer.eos_token_id
                                  )
    
    response.append(tokenizer.decode(sample_output[0][input_ids.shape[1]:], 
                                     skip_special_tokens=False).split('[EOR]')[0]
                   )

    i = i+1
    if i%10==0:
        print(i, "done out of ", len(test))

df = pd.DataFrame.from_dict({'Context':context, 'Actual':actual, 'Response':response})
df.to_csv('results-gpt2.csv', index=False)

