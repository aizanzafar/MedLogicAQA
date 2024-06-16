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
MODEL_NAME = "meta-llama/Llama-2-7b-hf"


# Fine-tuned model
new_model = "llama2-7b-with_rules"

######################### Model setup ##########################
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
        use_auth_token=access_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=access_token, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    return model, tokenizer

model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False
model.config.pretraining_tp = 1
print(model.config.quantization_config.to_dict())
######################### Model setup Done ##########################

######################### Dataset setup ##########################
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


with open('../../Data/data_withRule/train_4234.json','r') as f:
  data=json.load(f)

print(type(data))
print(len(data))

train_list, temp_list = train_test_split(data, test_size=0.30, random_state=42)
val_list, test_list = train_test_split(temp_list, test_size=0.25, random_state=42)  # 0.25 of the original size is 20% of the original


# Display the sets
print("Training set size:", len(train_list), type(train_list))
print("Validation set size:", len(val_list), type(val_list))
print("Test set size:", len(test_list), type(test_list))


filtered_train_data = [item for item in train_list if '###response: {}' not in item]
filtered_val_data = [item for item in val_list if '###response: {}' not in item]
filtered_test_data = [item for item in test_list if '###response: {}' not in item]

train_dict={"text": filtered_train_data}
data_train = Dataset.from_dict(train_dict)
print(data_train)
dataset_train = process_dataset(data_train)
# print(dataset_train[0])
print("size of training sample: ",len(dataset_train))

val_dict={"text": filtered_val_data}
data_val = Dataset.from_dict(val_dict)
print(data_val)
dataset_val = process_dataset(data_val)
# print(dataset_val[0])
print("size of Validation sample: ",len(dataset_val))

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

############################# Dataset setup Done ##############################
######################### training ##########################
lora_r = 64 # LoRA attention dimension
lora_alpha = 16 # Alpha parameter for LoRA scaling
lora_dropout = 0.1 # Dropout probability for LoRA layers

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
################################################################################
# TrainingArguments parameters
################################################################################
OUTPUT_DIR = "results_llama2_with_rule" # Output directory where the model predictions and checkpoints will be stored

training_arguments = TrainingArguments(
    per_device_train_batch_size=12,  # Batch size per GPU for training
    per_device_eval_batch_size = 6, # Batch size per GPU for evaluation
    gradient_accumulation_steps=2,  # Number of update steps to accumulate the gradients for
    gradient_checkpointing = True,  # Enable gradient checkpointing
    optim="paged_adamw_32bit",      # Optimizer to use
    logging_steps=25,               # Log every X updates steps
    learning_rate=2e-4,             # Initial learning rate (AdamW optimizer)
    fp16=False,                     # Enable fp16/bf16 training (set bf16 to True with an A100)
    bf16 = False,
    max_grad_norm=0.3,              # Maximum gradient normal (gradient clipping)
    num_train_epochs=3,             # Number of training epochs
    max_steps = -1,                 # Number of training steps (overrides num_train_epochs)
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.03,              # Ratio of steps for a linear warmup (from 0 to learning rate)
    weight_decay = 0.001,           # Weight decay to apply to all layers except bias/LayerNorm weights
    save_strategy="epoch",
    group_by_length=True,           # Group sequences into batches with same length Saves memory and speeds up training considerably
    output_dir=OUTPUT_DIR,
    # save_safetensors=True,
    lr_scheduler_type="cosine",     # Learning rate schedule
    seed=42,
)

# # Save checkpoint every X updates steps
# save_steps = 0

################################################################################
# SFT parameters
################################################################################

max_seq_length = 750 # Maximum sequence length to use
packing = False # Pack multiple short examples in the same input sequence to increase efficiency

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset= dataset_val,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

trainer.train()
# trainer.save_model()
# trainer.model

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)


print("######################### training done ##########################")
