from vgg import create_vgg
from resnet import create_resnet
from dataset import LoadSkinDiseaseD
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

resume = False # resume training
checkpoint = "D:\\Allen_2023\\CNN\\resnet18_model19.pth"
num_epochs = 500
learning_rate = 0.005
batch_size = 120
num_workers = 8
experiment_name = 'vgg19'
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
criterion = nn.CrossEntropyLoss()
image_path = "D:\Allen_2023\IMG_CLASSES"

if __name__ == '__main__':
    # need to move these into main class or else function will keep calling them because of num_workers which is problematic
    # https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection
    
    train_dataset = LoadSkinDiseaseD(image_path, split="train", indexes = "D:\Allen_2023\CNN\split_D.pickle", augment=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                    shuffle=True, pin_memory=True, drop_last=True, num_workers = num_workers, persistent_workers=True)

    val_dataset = LoadSkinDiseaseD(image_path, split="val", indexes = "D:\Allen_2023\CNN\split_D.pickle")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, pin_memory=True,
                                         num_workers = num_workers, persistent_workers=True)

    print(len(train_dataset))
    model = create_vgg('vgg19').to(device)

    if resume:
        model.load_state_dict(torch.load(checkpoint))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))  

    # Train the model
    print("Starting Training Loop...")

    best_acc = 0
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

            epoch_loss+= loss
        epoch_loss = epoch_loss.item()
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

                epoch_loss += loss
                _, predicted = torch.max(pred.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum()
        acc = 100.0*correct.item()/total
        val_acc.append(acc)
        epoch_loss = epoch_loss.item()
        print("val epoch loss", epoch, epoch_loss)
        val_loss.append(epoch_loss)
        print("Accuracy: ", acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), experiment_name + "_model" + str(epoch) + ".pth")

    plt.subplot(2, 1, 1)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.yscale('log')

    plt.subplot(2, 1, 2)
    plt.plot(val_acc)
    plt.show()
