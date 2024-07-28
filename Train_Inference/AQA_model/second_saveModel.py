import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from sklearn.model_selection import train_test_split
import json
from datasets import Dataset

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

base_model = "finetuned_m1"
# Fine-tuned model
new_model = "llama-2-7b-fol-kg-M2"

access_token = "" 


# Reload model in FP16 and merge it with LoRA weights
load_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
    # use_auth_token=access_token,
)

model = PeftModel.from_pretrained(load_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



print("saving model")
model.save_pretrained("finetuned_m2")
tokenizer.save_pretrained("finetuned_m2")

print("done")