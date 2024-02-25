import os
import random
import math
import pickle
import copy
import csv 

path = "D:\Allen_2023\LLM\Skin_Disease_Classes_v3.csv"

ratio = [0.8, 0.2]

total = 0

#### generate train test split
texts = {}
with open(path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        text = row[4:][0]
        target = row[0]
        if target in texts:
            texts[target].append(text)
        else:
            texts[target] = [text]

classes = {x:i for i, x in enumerate(set(texts.keys()))}

train_index = []
train_targets = []
val_index = []
val_targets = []
for target, text in texts.items():
    train = copy.deepcopy(random.sample(text, math.ceil(len(text)*ratio[0])))
    train_index+=(train)
    train_targets+=([target]*len(train))

    text = set(text)
    val = list(text - set(train))
    val_index+=(val)
    val_targets+=([target]*len(val))


print(len(train_index), len(val_index))
print((set(train_index).intersection(set(val_index))))

split = {
    'train': train_index,
    'train_labels': train_targets,
    'val': val_index,
    'val_labels': val_targets,
    'classes': classes
}

with open('split_descriptions_v3.pickle', 'wb') as handle:
    pickle.dump(split, handle, protocol=pickle.HIGHEST_PROTOCOL)


#### test splits
with open('split_descriptions_v3.pickle', 'rb') as handle:
    total = pickle.load(handle)
    
    
train = total["train"]
val = total["val"]
print(len(train), len(val))
print(len(set(train).intersection(set(val))))

