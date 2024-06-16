import json
import re
from pprint import pprint

import pandas as pd
import torch
from huggingface_hub import notebook_login
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer
from datasets import Dataset
from sklearn.model_selection import train_test_split

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=42)
        .map(generate_text)
    )

def generate_text(data_point):
    conversation_text = data_point['text']
    return {
        "text": conversation_text,
    }


with open('../Data/data_m2/train_4234.json','r') as f:
  data=json.load(f)

print(type(data))
print(len(data))

train_list, temp_list = train_test_split(data, test_size=0.30, random_state=42)
val_list, test_list = train_test_split(temp_list, test_size=0.25, random_state=42)  # 0.25 of the original size is 20% of the original



# Remove everything after '###response:'
new_list = [item.split('###response:')[0] + '###response: ' for item in test_list]

# Display the new list
print(len(new_list))

test_dict={"text": new_list}
data_test = Dataset.from_dict(test_dict)
print(data_test)
dataset_test = process_dataset(data_test)
# print(dataset_test[0])
print("size of testing sample: ",len(dataset_test))

######################### Model setup ##########################
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "finetuned_m1"

# Fine-tuned model
new_model = "llama-2-7b-fol-kg-M2"

def create_model_and_tokenizer():
    bnb_4bit_compute_dtype = "float16"
    use_4bit = True
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # Activate 4-bit precision base model loading
        bnb_4bit_quant_type="nf4",              # Quantization type (fp4 or nf4)
        bnb_4bit_compute_dtype=compute_dtype,   # Compute dtype for 4-bit base models
        bnb_4bit_use_double_quant=False,        # Activate nested quantization for 4-bit base models (double quantization)
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    return model, tokenizer

model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False
model.config.pretraining_tp = 1
# print(model.config.quantization_config.to_dict())
######################### Model setup Done ##########################

print('######################### Model setup Done ##########################')

print('######################### inference ##########################')

model = PeftModel.from_pretrained(model, new_model)

def summarize(model, text: str):
    inputs = tokenizer(text, return_tensors="pt",return_token_type_ids=False).to(DEVICE)
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0001)
    return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)


print("######################### 20 test dataset ##########################")
for index, item in enumerate(test_list):
        print("index no: ",index,"\n")
        test_case= item.split('###response:')[0]
        test_case = test_case + ' ###response:'
        print(test_case)
        print("######################### expected output ##########################")
        print(item.split('###response:')[1])
        # print("######################### inference method 1 ##########################")
        # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024)
        # result = pipe(f"{test_case}")
        # print(result[0]['generated_text'])
        print("######################### inference method 2 ##########################")
        summary = summarize(model, test_case)
        print(summary)
        if index >= 20:
            break



