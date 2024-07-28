import json
import os
import string

with open('mashqa_train_4000_data.json','r') as data:
	without_r=json.load(data)

with open('../../V2_MedEx/Data/2_hops/Final_2hops_MashQA_kg_train_data_with_rule.json','r') as data:
	with_r=json.load(data)

train_data=[]
count=0
for q1,q2 in zip(without_r,with_r):
	print("context no: ",count)
	qq=[]
	for qa1,qa2 in zip(without_r[count]['qas'],with_r[count]['qas']):
		qa1_text=qa1['question']
		qa2_text=qa2['question']
		if qa1_text==qa2_text:
			#print(qa1_text)
			q = {
				"id":qa2["id"],
				"is_impossible": qa2["is_impossible"],
				"question": qa2_text,
				"rule_1": qa2['rule_1'],
				"rule_2": qa2['rule_2'],
				"rule_3": qa2['rule_3'],
				"rule_4": qa2['rule_4'],
				"rule_5": qa2['rule_5'],
				"rule_6": qa2['rule_6'],
				"answers": qa1['answers'][0]
			}
			qq.append(q)
	train ={
			"context":q2["context"],
			"qas":qq
	}
	train_data.append(train)
	count=count+1

file_name='MashQA_train_data_with_rule.json'
print(file_name)
with open(file_name, 'w') as fp:
	json.dump(train_data, fp, indent=4)

