import os 
import torch
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
from tqdm import tqdm
import numpy as np
from PIL import Image

class LoadSkinDiseaseD(Dataset):
    def __init__(self, path, augment = False, size = 256):
        classes = os.walk(path)
        self.files = []
        self.classes = []
        self.labels = []

        self.augment = augment
        self.size = size
        self.transform = self.data_aug()

        for i, file in enumerate(classes):
            if i == 0:
                continue 
            label = os.path.split(file[0])[1]
            self.classes.append(label)
            for item in file[2]:
                self.files.append(os.path.join(file[0], item))
                self.labels.append(i-1)

    def data_aug(self):
        transform_list = []
        transform_list += [Transforms.ToTensor()]
        if self.augment:
            transform_list += [
                Transforms.ColorJitter(brightness=.5, contrast=.3, saturation=.3),
                Transforms.RandomHorizontalFlip(),
                Transforms.RandomVerticalFlip(),
            ]
        transform_list += [Transforms.Normalize(
                    mean=np.array([0.5, 0.5, 0.5]),
                    std=np.array([0.5, 0.5, 0.5]),
                )]
        custom_augmentation = Transforms.Compose(transform_list)
        return custom_augmentation

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        image = Image.open(file).convert('RGB')
        image = image.resize((self.size,self.size), resample=Image.Resampling.BICUBIC)
        image = np.array(image)

        label = self.labels[index]

        image = self.transform(image)

        return image, label

if __name__ == "__main__":
    image_path = "D:\Allen_2023\IMG_CLASSES"

    dataset = LoadSkinDiseaseD(image_path)
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=True, pin_memory=True, drop_last=True)
    
    print(len(dataset))
    # for i, data in tqdm(enumerate(dataloader)):
    #     print(i, data[0])