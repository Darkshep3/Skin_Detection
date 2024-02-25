import os
import random
import math
import pickle

image_path = "D:\Allen_2023\IMG_CLASSES"

ratio = [0.8, 0.2]

classes = os.walk(image_path)
count = 0

for i, file in enumerate(classes):
    if i == 0:
        continue 
    for item in file[2]:
        count+=1

train_index = random.sample([i for i in range(count)], int(count*ratio[0]))
val_index = random.sample([i for i in range(count)], math.ceil(count*ratio[1]))
# test_index = random.sample([i for i in range(count)], math.ceil(count*ratio[2]))

print(count, len(train_index), len(val_index))#, len(test_index))

split = {
    'train': train_index,
    'val': val_index,
    # 'test': test_index
}

with open('split_D.pickle', 'wb') as handle:
    pickle.dump(split, handle, protocol=pickle.HIGHEST_PROTOCOL)
