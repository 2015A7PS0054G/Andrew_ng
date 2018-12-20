import json
import os
from my_dict import my_dict

with open('./my_save_file', 'r') as fr:
	my_results = json.load(fr)
	all_names = [a[0] for a in my_results]

test_dir = "./my_test"
all_im = os.listdir(test_dir)
all_im = [int(res.split('.')[0]) for res in all_im]

TP = 0
FP = 0
FN = 0
total_score = 0
count = 0

for img_name in all_im:
	wanted = set([int(a) for a in my_dict[img_name]])
	result = set(my_results[all_names.index(str(img_name)+".png")][1])
	#print(count, wanted, result)
	TP = len(wanted.intersection(result))
	FN = len(wanted.difference(wanted.intersection(result)))
	FP = len(result.difference(wanted.intersection(result)))
	print(TP, FN, FP, " :: ",)
	score = float(TP)/(TP+FP+FN)
	print(score)
	total_score += score
	count += 1

print(total_score, count, total_score/count)
