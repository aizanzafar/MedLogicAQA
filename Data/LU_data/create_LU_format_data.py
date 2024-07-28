import json
import pandas as pd

with open('../mashqa_data/MashQA_train_data_with_rule.json','r') as f:
	data=json.load(f)

print("data loaded")

def remove_duplicates(input_list):
	# Convert inner lists to tuples for hashing (lists are unhashable)
	hashed_data = {tuple(item): True for item in input_list}

	# Convert back to list of lists
	unique_data = [list(item) for item in hashed_data.keys()]

	return unique_data

prompt = """Using the following rules, create a knowledge graph triple to support the answer for the given question and context.

### Rule: 
Rule of Co-occurrence: co_occurs_with(X, Y) ∧ affects(Y, Z) => affects(X, Z)
Rule of Prevention and Causation: prevent(X, Y) ∧ causes(Y, Z) => prevent(X, Z)
Rule of Treatment and Classification: treat(X, Y) ∧ is_a(Y, Z) => treat(X, Z)
Rule of Diagnosis and Interaction: diagnosis(X, Y) ∧ interacts_with(X, Z) => diagnosis(Z, Y)
Rule of Conjunction: co_occurs_with(X, Y) ∧ affects(X, Z) => co_occurs_with(Y, Z)
Rule of Disjunction: (prevent(X, Y) ∨ causes(Y, Z)) => (prevent(X, Z) ∨ causes(X, Z))
"""

data_train=[]

for item in data:
	context= "###context: "+item['context']
	for qa in item['qas']:
		response={}
		ques="###question: "+qa['question']
		ans="###answer: "+qa['answers']
		if len(remove_duplicates(qa['rule_1']))==1:
			pass
		else:
			response['Rule of Co-occurrence']= qa['rule_1']
		if len(remove_duplicates(qa['rule_2']))==1:
			pass
		else:
			response['Rule of Prevention and Causation']= qa['rule_2']
		if len(remove_duplicates(qa['rule_3']))==1:
			pass
		else:
			response['Rule of Treatment and Classification']= qa['rule_3']
		if len(remove_duplicates(qa['rule_4']))==1:
			pass
		else:
			response['Rule of Diagnosis and Interaction']= qa['rule_4']
		if len(remove_duplicates(qa['rule_5']))==1:
			pass
		else:
			response['Rule of Conjunction']= qa['rule_5']
		if len(remove_duplicates(qa['rule_6']))==1:
			pass
		else:
			response['Rule of Disjunction']= qa['rule_6']
		text = prompt + "\n"+context + "\n" + ques + "\n" + ans +"\n###response: " + str(response)
		# print(text)
		data_train.append(text)

file_name='mashqa_val_data_with_rule.json'
print(file_name)
with open(file_name, 'w') as fp:
	json.dump(data_train, fp, indent=4)
