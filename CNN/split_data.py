import os
import random
import math
import pickle
import copy
from PIL import Image 
import numpy as np

image_path = "D:\Allen_2023\IMG_CLASSES_FINAL"

ratio = [0.8, 0.2]

classes = os.walk(image_path)
total = 0

### renaming function
# for i, file in enumerate(classes):
#     if i == 0:
#         continue 
#     for name in file[2]:
#         val = file[0].split("\\")[-1]
#         src = os.path.join(image_path, val, name)
#         dst = os.path.join(image_path, val, val + "_" + name)
#         os.rename(src, dst)
#         total += 1
# print(total)


#### generate train test split
# train_index = []
# val_index = []
# for i, file in enumerate(classes):
#     if i == 0:
#         continue 
#     count = len(file[2])
#     total += count
#     train = copy.deepcopy(random.sample(file[2], math.floor(count*ratio[0])))
#     train_index += train 
#     for i in train:
#         file[2].remove(i)
#     val_index += copy.deepcopy(file[2])

# print(len(train_index), len(val_index))
# print((set(train_index).intersection(set(val_index))))

# split = {
#     'train': train_index,
#     'val': val_index,
# }

# with open('split_D_updated.pickle', 'wb') as handle:
#     pickle.dump(split, handle, protocol=pickle.HIGHEST_PROTOCOL)


#### test splits
with open('split_final.pickle', 'rb') as handle:
    total = pickle.load(handle)
    
    
train = total["train"]
val = total["val"]
print(len(train), len(val))
print(len(set(train).intersection(set(val))))

#### evaluate dataset
# img_size = set()
# for i, file in enumerate(classes):
#     if i == 0:
#         continue 
#     for name in file[2]:
#         val = file[0].split("\\")[-1]
#         src = os.path.join(image_path, val, name)
#         a = np.array(Image.open(src))
#         img_size.add(tuple(a.shape))

# print(img_size)