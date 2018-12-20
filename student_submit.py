import json
import model
import os
import sys
from time import time

startTime = time()

test_image_directory = "./my_test/"
all_im = os.listdir(test_image_directory)

output_file = open("predict_output", "w")
my_results = []

for img in all_im:
    oneTime = time()
    print(img,)
    #sys.stdout = output_file
    result = model.predict(test_image_directory+img)
    #sys.stdout = sys.__stdout__
    my_results.append([img, [int(a) for a in result]])
    print(result, time()-oneTime)

print(time()-startTime, "ALL DONE") 
with open('my_save_file', 'w') as fw:
    fw.write(json.dumps(my_results))
