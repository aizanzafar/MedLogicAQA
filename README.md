# MedLogicAQA Experiment

This README file provides step-by-step instructions for reproducing the MedLogicAQA experiment. It covers data creation, training, and inference processes.

## Data Creation

### 1. Converting Extractive MASHQA Dataset into Abstractive Format
To convert the MASHQA dataset into an abstractive format, follow these steps:
1. Run the script located at `Data/mashqa_data/convert_mashqa_data.py`.
2. Save the output file as `mashqa_train_data.json`.

### 2. KG Construction Using QUICK-UMLS
To construct the knowledge graph (KG) using QUICK-UMLS, follow these steps:
1. Run the script located at `Data/UMLS_KG_Preprocess/final_preprocess.py`.
2. Save the output file as `Data/mashqa_data/MashQA_kg_train_data.json`.

### 3. Integrate RULE with MashQA KG Train Data
To integrate rules with the MashQA KG train data, follow these steps:
1. Run the script located at `Data/Rule_integrate/cosine_triple.py`. This will save the file as `mashqa_data_withRule`.
2. Run the script `Data/data_preparation.py` to create a file named `MashQA_train_data_with_rule.json`.

### 4. Create LU Format Data
To create data in the LU format, follow these steps:
1. Run the script located at `Data/LU_data/create_LU_format_data.py`.
2. Save the output file as `train_LU_model.json`.

### 5. Create AQA Format Data
To create data in the AQA format, follow these steps:
1. Run the script located at `Data/AQA_data/create_AQA_format_data.py`.

## Training and Inference

### LU Model

#### Training
To train the LU model, follow these steps:
1. Run the script located at `Train_Inference/LU_model/train_lu_model.py`.
2. The script will read the file `Data/LU_data/train_LU_model.json` prepared earlier.
3. The model predictions and checkpoints will be stored in the directory specified by `OUTPUT_DIR = "results_LU_model"`.

#### Inference
To perform inference with the LU model, follow these steps:
1. Run the script located at `Train_Inference/LU_model/inference_aa.py`.

### AQA Model

#### Training
To train the AQA model, follow these steps:
1. Run the script located at `Train_inference/AQA_model/train_aqa_model.py`.

### Hyperparameters
Information about the hyperparameters used in the training process can be found in Appendix F of the accompanying documentation.



---
This README file aims to provide clear and comprehensive instructions to facilitate the replication of the MedLogicAQA experiment.
