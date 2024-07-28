from nlgeval import compute_metrics
from evaluate import load
from numpy import mean

def eval(model_name):
    path = 'results/'+model_name+'.txt'

    with open(path, 'r') as hypothesis:
        hypothesis = hypothesis.read().split('\n')

    with open('results/gold.txt', 'r') as references:
        references = references.read().split('\n')


    bertscore = load("bertscore")
    results = bertscore.compute(predictions=hypothesis, references=references, lang="en")

    # print("Bert Scores are: \n", results)
    print("precision = ", mean(results['precision']))
    print("recall = ", mean(results['precision']))
    print("f1 = ", mean(results['f1']))


    metrics_dict = compute_metrics(hypothesis=path,
                                   references=['./results/gold.txt'])



# print("Results for gpt2")
# eval('gpt2')
# print("###############################################################")

# print("Results for summ")
# eval('gpt2sum')
# print("###############################################################")

# print("Results for persona")
# eval('gpt2per')
# print("###############################################################")

# print("Results for rl")
# eval('gpt2rl')
# print("###############################################################")

print("Results for Influencer")
eval('influ')
print("###############################################################")

