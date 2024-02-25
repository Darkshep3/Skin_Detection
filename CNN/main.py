from vgg import create_vgg
from resnet import create_resnet
from efficientnet import create_effnet
from vit import create_vit
from dataset import LoadSkinDiseaseFinal
from torch.utils.data import WeightedRandomSampler, DataLoader
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

resume = False # resume training
checkpoint = "D:\\Allen_2023\\CNN\\resnet50_0_224_model64.pth"
num_epochs = 500
learning_rate = 0.005
batch_size = 120
num_workers = 8
experiment_name = 'vit_0'
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
criterion = nn.CrossEntropyLoss()
image_path = "D:\Allen_2023\IMG_CLASSES_FINAL"
pretrained = False
torch.hub.set_dir("D:\Allen_2023\model_weights")

if __name__ == '__main__':
    # need to move these into main class or else function will keep calling them because of num_workers which is problematic
    # https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection
    
    train_dataset = LoadSkinDiseaseFinal(image_path, split="train", indexes = "D:\Allen_2023\CNN\split_final.pickle",
                                         augment=True, size = 224)
    sampler = WeightedRandomSampler(weights= train_dataset.gen_sampler(), num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, pin_memory=True,
                              num_workers = num_workers, persistent_workers=True)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size,
    #                 shuffle=True, pin_memory=True, drop_last=True, num_workers = num_workers, persistent_workers=True)

    val_dataset = LoadSkinDiseaseFinal(image_path, split="val", indexes = "D:\Allen_2023\CNN\split_final.pickle", size = 224)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True,
                                         num_workers = num_workers, persistent_workers=True)

    print(len(train_dataset))
    model = create_vit(num_classes = 26).to(device)

    if resume:
        model.load_state_dict(torch.load(checkpoint))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))  

    # Train the model
    print("Starting Training Loop...")

    best_acc = 0
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        correct = 0
        total = 0
        model.train()
        if pretrained:
            model.freeze_weights()
        for i, data in enumerate(train_loader):
            image, label = data 
            image = image.to(device)
            label = label.to(device)

            pred = model(image)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+= loss
            _, predicted = torch.max(pred.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum()
        acc = 100.0*correct.item()/total
        train_acc.append(acc)   
        epoch_loss = epoch_loss.item()
        train_loss.append(epoch_loss)
        print("train epoch loss", epoch, epoch_loss)
        print("Train Accuracy: ", acc)

        epoch_loss = 0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device)
                label = label.to(device)

                pred = model(image)
                loss = criterion(pred, label)

                epoch_loss += loss
                _, predicted = torch.max(pred.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum()
        acc = 100.0*correct.item()/total
        val_acc.append(acc)
        epoch_loss = epoch_loss.item()
        print("val epoch loss", epoch, epoch_loss)
        val_loss.append(epoch_loss)
        print("Val Accuracy: ", acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), experiment_name + "_model" + str(epoch) + ".pth")

    plt.subplot(3, 1, 1)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.yscale('log')

    plt.subplot(3, 1, 2)
    plt.plot(train_acc)

    plt.subplot(3, 1, 3)
    plt.plot(val_acc)
    plt.show()
