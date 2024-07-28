import json
import os
import string
import sys
from io import open

import numpy as np

from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel

"""
# 1. rule of co-coocurance
co_occurs_with(X, Y) ∧ affects(Y, Z) => affects(X, Z)
#  2. Rule of Prevention and Causation:
prevent(X, Y) ∧ causes(Y, Z) => prevent(X, Z)
# 3. Rule of Treatment and Classification:
treat(X, Y) ∧ is_a(Y, Z) => treat(X, Z)
# 4. Rule of Diagnosis and Interaction:
diagnosis(X, Y) ∧ interacts_with(X, Z) => diagnosis(Z, Y)
# 5. Rule of Conjunction:
co_occurs_with(X, Y) ∧ affects(X, Z) => co_occurs_with(Y, Z)
# 6. Rule of Disjunction:
(prevent(X, Y) ∨ causes(Y, Z)) => (prevent(X, Z) ∨ causes(X, Z))

"""

def remove_duplicate(kg_triple):
	res = []
	[res.append(x) for x in kg_triple if x not in res]
	return res


def parse_triple(kg_triplets):
	kg_len = len(kg_triplets)
	empty=['_NAF_H','_NAF_R','_NAF_O']
	if kg_len <=20:
		tt= 20 - kg_len
		for item in range(tt):
			kg_triplets.append(empty)
	return kg_triplets


def apply_rules_to_kg(kg_triplets):
	co_occurs_triplets = []
	prevent_triplets = []
	treatment_triplets = []
	diagnosis_triplets = []
	conjunction_triplets = []
	disjunction_triplets = []
	# 1.Rule of Co-occurrence: If X co-occurs with Y and Y affects Z, then X affects Z
	for triplet in kg_triplets:
		if triplet[1] == "co-occurs_with":
			for other_triplet in kg_triplets:
				if other_triplet[0] == triplet[2] and other_triplet[1] == "affects":
					if triplet[0] == other_triplet[2]:
						pass
					else:
						co_occurs_triplets.append([triplet[0], "affects", other_triplet[2]])

	# 2.Rule of Prevention and Causation: If X prevents Y and Y causes Z, then X prevents Z
	for triplet in kg_triplets:
		if triplet[1] == "prevents":
			####commented out purposly
	# 3.Rule of Treatment and Classification: If X treats Y and Y is a type of Z, then X can be used to treat Z
	for triplet in kg_triplets:
		if triplet[1] == "treats":
			###### commented out

	# 4.Rule of Diagnosis and Interaction: If X is diagnosed with Y and X interacts with Z, then Z can be used for the diagnosis of Y
	for triplet in kg_triplets:
		if triplet[1] == "diagnoses":
			#### commented out
	# 5.Rule of Conjunction .
	for triplet in kg_triplets:
		if triplet[1] == "co-occurs_with":
			for other_triplet in kg_triplets:
				if other_triplet[0] == triplet[0] and other_triplet[1] == "affects":
					if triplet[2] == other_triplet[2]:
						pass
					else:
						conjunction_triplets.append([triplet[2], "co-occurs_with", other_triplet[2]])
	# 5.Rule of disjunction .
	for triple in kg_triplets:
		### commented out

	return parse_triple(remove_duplicate(co_occurs_triplets)),parse_triple(remove_duplicate(prevent_triplets)),parse_triple(remove_duplicate(treatment_triplets)),parse_triple(remove_duplicate(diagnosis_triplets)),parse_triple(remove_duplicate(conjunction_triplets)),parse_triple(remove_duplicate(disjunction_triplets))

def get_bert_based_similarity(sentence_pairs, model, tokenizer):
    """
    computes the embeddings of each sentence and its similarity with its corresponding pair
    Args:
        sentence_pairs(dict): dictionary of lists with the similarity type as key and a list of two sentences as value
        model: the language model
        tokenizer: the tokenizer to consider for the computation    
    Returns:
        similarities(dict): dictionary with similarity type as key and the similarity measure as value
    """
    similarities = dict()
    inputs_1 = tokenizer(sentence_pairs[0], return_tensors='pt')
    sent_1_embed = np.mean(model(**inputs_1).last_hidden_state[0].detach().numpy(), axis=0)

    for count,triple in enumerate(sentence_pairs[1]):
        # print(count)
        ## commented out
    return similarities

def sorted_triple(question,rule_kg_triples,pubmed_bert_model,pubmed_bert_tokenizer):	
	sentence_pairs = [question,rule_kg_triples]
	sim_final_dict= get_bert_based_similarity(sentence_pairs, pubmed_bert_model, pubmed_bert_tokenizer)

	triple_n_sim_final_dic=dict()
	for k,t in zip(sim_final_dict.keys(),final_kg_triplet):
		triple_n_sim_final_dic[k]=t

	sorted_dict = {k: v for k, v in sorted(sim_final_dict.items(), key=lambda item: item[1], reverse=True)}

	sorted_triple_list=[]
	for k,v in sorted_dict.items():
		sorted_triple_list.append(triple_n_sim_final_dic[k])

	return sorted_triple_list

## read mashqa-data
with open('mashqa_data/mashqa_train_data.json','r') as f:
	data=json.load(f)

print("data loaded")


pubmed_bert_model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
pubmed_bert_tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

train_data=[]

for num,item in enumerate(data):
	print("sequence number: ",num)
	question= item['qas'][0]['question']
	answer = item['qas'][0]['answer']
	r1,r2,r3,r4,r5,r6 = apply_rules_to_kg(item['kg triple'])
	r1_sorted = sorted_triple(question, r1, pubmed_bert_model, pubmed_bert_tokenizer)
	r2_sorted = sorted_triple(question, r2, pubmed_bert_model, pubmed_bert_tokenizer)
	r3_sorted = sorted_triple(question, r3, pubmed_bert_model, pubmed_bert_tokenizer)
	r4_sorted = sorted_triple(question, r4, pubmed_bert_model, pubmed_bert_tokenizer)
	r5_sorted = sorted_triple(question, r5, pubmed_bert_model, pubmed_bert_tokenizer)
	r6_sorted = sorted_triple(question, r6, pubmed_bert_model, pubmed_bert_tokenizer)
	q = {
	"id":num,
	"context": item['context'],
	"question": question,
	"kg_triples": item['kg triple'],
	"rule_1": r1_sorted,
	"rule_2": r2_sorted,
	"rule_3": r3_sorted,
	"rule_4": r4_sorted,
	"rule_5": r5_sorted,
	"rule_6": r6_sorted,
	"answer": answer
	}
	train_data.append(q)
	if len(train_data)==10:
		file_name='bioasq_train_'+str(len(train_data))+'.json'
		print(file_name)
		with open(file_name, 'w') as fp:
			json.dump(train_data, fp, indent=4)


file_name='mashqa_data_withRule'+str(len(train_data))+'.json'
print(file_name)
with open(file_name, 'w') as fp:
	json.dump(train_data, fp, indent=4)


