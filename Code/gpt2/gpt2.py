from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

from datasets import load_dataset



dataset = load_dataset('csv', data_files={'train': 'train_data.csv', 'eval':'val_data.csv', 'test': 'test_data.csv'})


#===============================================Format Data===============================================================

model_checkpoint = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

special_tokens_dict = {'additional_special_tokens': ['[EOC]','[SOC]','[SOR]', '[EOR]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    return tokenizer(examples["text"])

# block_size = tokenizer.model_max_length

block_size=512

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

# tokenizer.decode(lm_dataset["train"][1]["input_ids"])

#===============================================Train Model===============================================================


model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    f"{model_name}-bioasq",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["eval"],
)

trainer.train()
trainer.save_model()
