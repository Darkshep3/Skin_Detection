import os 
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms.v2 as Transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import numpy as np
from PIL import Image
import pickle
from matplotlib import pyplot as plt

class LoadSkinDiseaseD(Dataset):
    def __init__(self, path, split = "train", indexes = None, augment = False, size = 256):
        classes = os.walk(path)
        self.files = []
        self.classes = {}
        self.labels = []

        self.augment = augment
        self.size = size
        self.transform = self.data_aug()
        self.split = split 
        self.indexes = indexes
    
        for i, file in enumerate(classes):
            if i == 0:
                continue 
            label = os.path.split(file[0])[1]
            self.classes[i-1] = label
            for item in file[2]:
                self.files.append(os.path.join(file[0], item))
                self.labels.append(i-1)

        if indexes: 
            with open(indexes, 'rb') as handle:
                subset = pickle.load(handle)[split]
            temp_files = []
            temp_labels = []
            for val in subset:
                temp_files.append(self.files[val])
                temp_labels.append(self.labels[val])
            self.files = temp_files 
            self.labels = temp_labels


    def data_aug(self):
        transform_list = []
        transform_list += [Transforms.ToImage(),
                            Transforms.ToDtype(torch.uint8, scale=True)]
        if self.augment:
            transform_list += [
                Transforms.ColorJitter(brightness=.3, contrast=.2, saturation=.2),
                Transforms.RandomHorizontalFlip(),
                Transforms.RandomVerticalFlip(),
                # Transforms.RandomRotation(89),
                Transforms.GaussianBlur(7)
            ]
        transform_list += [
            Transforms.Resize(size = (self.size, self.size), antialias=True, 
                                interpolation = InterpolationMode.BICUBIC),
            Transforms.ToDtype(torch.float32, scale=True),
                                Transforms.Normalize(
                                mean=np.array([0.485, 0.456, 0.406]),
                                std=np.array([0.229, 0.224, 0.225]),
                )]
        custom_augmentation = Transforms.Compose(transform_list)
        return custom_augmentation

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        image = Image.open(file).convert('RGB')
        image = np.array(image)

        label = self.labels[index]

        image = self.transform(image)

        return image, label
    
class LoadSkinDiseaseFinal(Dataset):
    def __init__(self, path, split = "train", indexes = None, augment = False, size = 256):
        classes = os.walk(path)
        self.files = []
        self.classes = {}
        self.labels = []

        self.augment = augment
        self.size = size
        self.transform = self.data_aug()
        self.split = split 
        self.indexes = indexes
        subset = []

        if indexes: 
            with open(indexes, 'rb') as handle:
                subset = pickle.load(handle)[split]

        for i, file in enumerate(classes):
            if i == 0:
                continue 
            label = os.path.split(file[0])[1]
            self.classes[i-1] = label
            for item in file[2]:
                if indexes:
                    if item in subset:
                        self.files.append(os.path.join(file[0], item))
                        self.labels.append(i-1)
                else: 
                    self.files.append(os.path.join(file[0], item))
                    self.labels.append(i-1)
            
    def data_aug(self):
        transform_list = []
        transform_list += [Transforms.ToImage(),
                            Transforms.ToDtype(torch.uint8, scale=True)]
        if self.augment:
            transform_list += [
                Transforms.ColorJitter(brightness=.3, contrast=.2, saturation=.2),
                Transforms.RandomHorizontalFlip(),
                Transforms.RandomVerticalFlip(),
                Transforms.GaussianBlur(7),
                Transforms.Resize(size = (240, 240), antialias=True, 
                                interpolation = InterpolationMode.BICUBIC),
                Transforms.RandomRotation(89),
                Transforms.RandomCrop(size = (self.size, self.size)),
            ]
        else:
            transform_list += [
                Transforms.Resize(size = (self.size, self.size), antialias=True, 
                                interpolation = InterpolationMode.BICUBIC),
            ]
        transform_list += [
            Transforms.ToDtype(torch.float32, scale=True),
            Transforms.Normalize(
                                mean=np.array([0.485, 0.456, 0.406]),
                                std=np.array([0.229, 0.224, 0.225]),
                )
        ]
        custom_augmentation = Transforms.Compose(transform_list)
        return custom_augmentation
    
    def gen_sampler(self):
        freq = [0]*len(self.classes)
        for i in self.labels:
            freq[i]+=1
        for i in range(len(freq)):
            freq[i] = 1./ freq[i]
        probs = [freq[i] for i in self.labels]
        return probs

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        image = Image.open(file).convert('RGB')
        image = np.array(image)

        label = self.labels[index]

        image = self.transform(image)

        return image, label

if __name__ == "__main__":
    image_path = "D:\Allen_2023\IMG_CLASSES"

    # dataset = LoadSkinDiseaseFinal(image_path)
    # print(len(dataset))

    # dataset = LoadSkinDiseaseFinal(image_path, split="train", indexes = "D:\Allen_2023\CNN\split_D_updated.pickle", augment=True)
    # print(len(dataset))

    dataset = LoadSkinDiseaseFinal(image_path, split="val", indexes = "D:\Allen_2023\CNN\split_D_updated.pickle", augment=True, size =224)
    print(len(dataset))
    

    # batch_size = 100
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                 shuffle=True, pin_memory=True, drop_last=True)

    # for i, data in tqdm(enumerate(dataloader)):
    #     print(i, torch.max(data[0]), torch.min(data[0]))

    # sampler = WeightedRandomSampler(weights= dataset.gen_sampler(), num_samples=batch_size, replacement=True)
    # dataloader = torch.utils.data.DataLoader(dataset, sampler = sampler, pin_memory=True)

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #             shuffle=False, pin_memory=True, drop_last=True)

    # for i, data in tqdm(enumerate(dataloader)):
    #     print(i, torch.max(data[0]), torch.min(data[0]))
    #     plt.imshow(data[0][0].permute(1, 2, 0), interpolation='nearest')
    #     plt.show()
    