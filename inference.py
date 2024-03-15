from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from LLM.dataset import SkinDiseaseDataset as SkinDiseaseTextDataset, WINDOW
from CNN.dataset import LoadSkinDiseaseFinal as SkinDiseaseImageDataset
from CNN.resnet import create_resnet
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
batch_size = 1
SAMPLES = 5

path = "D:\Allen_2023\LLM\split_descriptions_v4.pickle"

cnn_checkpoint = "D:\\Allen_2023\\final_models\\dataset_final\\Resolution\\O1_146.pth"

llm_model_id = "D:\Allen_2023\model_weights\llama-7b"
cache_dir="D:\Allen_2023\model_weights"
llm_checkpoint = 'D:\Allen_2023\LLM\llama-lora-pred5-coo'

image_path = "D:\Allen_2023\IMG_CLASSES_FINAL"
path = "D:\Allen_2023\LLM\split_descriptions_v4.pickle"
torch.hub.set_dir(cache_dir)

elimination = True

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    val_dataset = SkinDiseaseImageDataset(image_path, split="val", indexes = "D:\Allen_2023\CNN\split_final.pickle", size = 300)
    image_class_map_inv = val_dataset.classes

    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    text_dataset = SkinDiseaseTextDataset(path, tokenizer, split="val", options = False, top = False, elimination = elimination)
    text_class_map = text_dataset.classes
    text_class_map_inv = dict((v,k) for k,v in text_dataset.classes.items())

    image_model = create_resnet('resnet50', num_classes = 26, pretrained='resnet50').to(device)
    image_model.load_state_dict(torch.load(cnn_checkpoint))


    text_model = AutoModelForSequenceClassification.from_pretrained(llm_model_id, 
                                                cache_dir=cache_dir,
                                                torch_dtype=torch.bfloat16,
                                                num_labels=26).to(device)
    text_model.config.pad_token_id = text_model.config.eos_token_id

    model = PeftModel.from_pretrained(text_model, llm_checkpoint, is_trainable=False)
    model = model.to(device)
    model.eval()


    print("Starting Evaluation...")

    correct = 0
    total = 0
    image_model.eval()

    pred_list = []
    target_list = []

    with torch.no_grad():
        for image, label in tqdm(val_loader):
            image = image.to(device)
            label = label.to(device)
            
            # get a corresponding text 
            image_label = int(label)
            if image_class_map_inv[image_label] == "Hair":
                text_label = text_class_map["Alopecia"]
            else: 
                text_label = text_class_map[image_class_map_inv[image_label]]
            label_index = text_dataset.search(text_label)
            index = random.choice(label_index)

            item = {}
            for key, val in text_dataset.encodings.items():
                item[key] = copy.deepcopy(val[index])
            item['labels'] = torch.tensor([text_label])


            # predict from image
            pred = image_model(image)

            top_3 = torch.argsort(pred.data, 1, descending=True)[:,:SAMPLES]
            top_3 = [image_class_map_inv[i] for i in top_3.tolist()[0]]
        
            # add top choices to text
            token = " Top Choices: " + str(top_3)
            token = tokenizer(token, padding=True)
            key_val = token['input_ids']
            key_attn = token['attention_mask']

            for key, val in item.items():
                if key == 'input_ids':
                    item['input_ids'] += key_val
                elif key == 'attention_mask':
                    item['attention_mask'] += key_attn

            item = {key: torch.tensor([val]) for key, val in item.items()}

            # elimination
            if elimination:
                initial_path = [list(text_dataset.classes.keys())] # batch size
                while len(initial_path[0]) > WINDOW:
                    input_ids = []
                    attn_mask = []
                    for index, path in enumerate(initial_path):
                        path = " Available Choices: [" +  (", ".join(path)) + "]"
                        path = text_dataset.to_token(path)
                        input_ids.append(torch.unsqueeze(torch.cat((item['input_ids'][index], torch.tensor(path[0]))),0))
                        attn_mask.append(torch.unsqueeze(torch.cat((item['attention_mask'][index], torch.tensor(path[1]))),0))

                    input_ids = torch.cat(input_ids,0).to(device)
                    attention_mask = torch.cat(attn_mask,0).to(device)
                    labels = item['labels'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                    new_path = []
                    for index, pred in enumerate(outputs[1].data):
                        to_delete = initial_path[index]
                        eliminated = 0
                        top_n = torch.argsort(pred, descending=False)
                        for i in top_n:
                            if text_class_map_inv[int(i)] in to_delete:
                                to_delete.remove(text_class_map_inv[int(i)])
                                eliminated += 1
                            if eliminated >= WINDOW:
                                break
                        new_path.append(to_delete)
                    initial_path = copy.deepcopy(new_path)        
            else:
                input_ids = item['input_ids'].to(device)
                attention_mask = item['attention_mask'].to(device)
                labels = item['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            _, predicted = torch.max(outputs[1].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            pred_list += predicted.flatten().tolist()
            target_list += labels.flatten().tolist()

    acc = 100.0*correct.item()/total
    print("Val Accuracy: ", acc)

    cm = confusion_matrix(pred_list, target_list)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # ConfusionMatrixDisplay(cm, display_labels=text_class_map.keys()).plot(xticks_rotation='vertical', colorbar=False)
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=val_dataset.classes.values(), yticklabels=val_dataset.classes.values(), cmap = 'viridis')
    plt.show()