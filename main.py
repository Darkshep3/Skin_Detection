from vgg import create_vgg
from dataset import LoadSkinDiseaseD
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

num_epochs = 150
learning_rate = 0.005
batch_size = 80
num_workers = 6

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
model = create_vgg('vgg19').to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))  

image_path = "D:\Allen_2023\IMG_CLASSES"

train_dataset = LoadSkinDiseaseD(image_path, split="train", indexes = "D:\Allen_2023\CNN\split_D.pickle", augment=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                shuffle=True, pin_memory=True, drop_last=True, num_workers = num_workers)

val_dataset = LoadSkinDiseaseD(image_path, split="val", indexes = "D:\Allen_2023\CNN\split_D.pickle")
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, pin_memory=True,
                                         num_workers = num_workers)

if __name__ == '__main__':
    print(len(train_dataset))

    # Train the model
    total_step = len(train_loader)
    print("Starting Training Loop...")

    train_loss = []
    val_loss = []
    val_acc = []
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            image, label = data 
            image = image.to(device)
            label = label.to(device)

            pred = model(image)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+= loss.item()
        train_loss.append(epoch_loss)
        print("train epoch loss", epoch, epoch_loss)

        epoch_loss = 0
        with torch.no_grad():
            correct = 0
            total = 0
            
            for image, label in val_loader:
                image = image.to(device)
                label = label.to(device)

                pred = model(image)
                loss = criterion(pred, label)

                epoch_loss += loss.item()
                _, predicted = torch.max(pred.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        val_acc.append(100.0*correct/total)
        print("val epoch loss", epoch, epoch_loss)
        val_loss.append(epoch_loss)
        print("Accuracy: ", 100.0*correct/total)

    plt.subplot(2, 1, 1)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.yscale('log')

    plt.subplot(2, 1, 2)
    plt.plot(val_acc)
    plt.show()