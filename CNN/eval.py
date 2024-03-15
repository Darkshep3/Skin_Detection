from vgg import create_vgg
from resnet import create_resnet
from efficientnet import create_effnet
from vit import create_vit
from dataset import LoadSkinDiseaseFinal
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np

checkpoint = "D:\\Allen_2023\\final_models\\dataset_final\\Resolution\\O1_146.pth"
batch_size = 150
num_workers = 8
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
criterion = nn.CrossEntropyLoss()
image_path = "D:\Allen_2023\IMG_CLASSES_FINAL"
torch.hub.set_dir("D:\Allen_2023\model_weights")
TOP = 5

if __name__ == '__main__':
    val_dataset = LoadSkinDiseaseFinal(image_path, split="val", indexes = "D:\Allen_2023\CNN\split_final.pickle", size = 300)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True,
                                         num_workers = num_workers, persistent_workers=True)

    print(len(val_dataset))
    # model = create_vgg('vgg11', num_classes = 26, pretrained='vgg11').to(device)
    model = create_resnet('resnet50', num_classes = 26, pretrained='resnet50').to(device)
    # model = create_effnet('efficientnet-b0', num_classes = 26, pretrained='efficientnet-b0').to(device)
    # model = create_vit(pretrained='vit', num_classes = 26).to(device)
    model.load_state_dict(torch.load(checkpoint))

    # Evaluate the model
    print("Starting Evaluation...")

    epoch_loss = 0
    correct = 0
    total = 0
    model.eval()

    pred_list = []
    target_list = []

    top_3_correct = 0
    with torch.no_grad():
        for image, label in tqdm(val_loader):
            image = image.to(device)
            label = label.to(device)

            pred = model(image)

            top_3 = torch.argsort(pred.data, 1, descending=True)[:,:TOP]
            # top_3 = 25-top_3 # uncomment if trained on linux


            for i in range(top_3.shape[0]):
                if label[i] in top_3[i]:
                    top_3_correct += 1

            loss = criterion(pred, label)
            epoch_loss += loss
            _, predicted = torch.max(pred.data, 1)
            total += label.size(0)
            # predicted = 25-predicted # uncomment if trained on linux


            correct += (predicted == label).sum()

            pred_list += predicted.flatten().tolist()
            target_list += label.flatten().tolist()

    acc = 100.0*correct.item()/total
    acc_3 = 100.0*top_3_correct/total
    epoch_loss = epoch_loss.item()
    print("Val Epoch Loss", epoch_loss)
    print("Val Accuracy: ", acc)
    print("Top 3 Accuracy: ", acc_3)

    cm = confusion_matrix(pred_list, target_list)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # ConfusionMatrixDisplay(cm, display_labels=val_dataset.classes.values()).plot(xticks_rotation='vertical', colorbar=False)
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=val_dataset.classes.values(), yticklabels=val_dataset.classes.values(), cmap = 'viridis')
    plt.show()