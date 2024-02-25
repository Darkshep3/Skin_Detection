import os
import hashlib
import random
import math
import pickle

image_path = "D:\Allen_2023\IMG_CLASSES"

ratio = [0.8, 0.2]

classes = os.walk(image_path)
hashes = {} # hash, file
count = 0

for i, file in enumerate(classes):
    if i == 0:
        continue 
    for item in file[2]:
        count+=1
        image = os.path.join(file[0], item)
        hash = None
        with open(image, "rb") as f:
            hash = hashlib.sha256(f.read()).hexdigest()
        if hash in hashes:
            os.remove(image) 
            count+=1
            print(hashes[hash], image)
        else:
            hashes[hash] = image

print(count)