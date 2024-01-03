from vgg import create_vgg
from dataset import LoadSkinDiseaseD
import torch
import torch.nn as nn
from tqdm import tqdm

num_epochs = 100
learning_rate = 0.005
batch_size = 120

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
model = create_vgg('vgg11').to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))  

image_path = "D:\Allen_2023\IMG_CLASSES"

dataset = LoadSkinDiseaseD(image_path)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                shuffle=True, pin_memory=True, drop_last=True)

print(len(dataset))

# Train the model
total_step = len(train_loader)
print("Starting Training Loop...")

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

        if i%20==0: 
            print(epoch, i, loss.item())
        epoch_loss+= loss.item()
    print("epoch loss")
    print(epoch, epoch_loss)