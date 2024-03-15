from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataset import SkinDiseaseDataset, WINDOW
import torch
import torch.nn as nn
from tqdm import tqdm
import copy

batch_size = 2
model_id = "D:\Allen_2023\model_weights\llama-7b"
cache_dir="D:\Allen_2023\model_weights"
path = "D:\Allen_2023\LLM\split_descriptions_v4.pickle"
experiment_name = 'D:\Allen_2023\LLM\llama-lora-pred3-COO'
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

criterion = nn.CrossEntropyLoss()

options = False 
top = True
elimination = True

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    val_dataset = SkinDiseaseDataset(path, tokenizer, split="val", options = options, top = top, elimination = elimination)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                    pin_memory=True)
    
    class_map = dict((v,k) for k,v in val_dataset.classes.items())

    model = AutoModelForSequenceClassification.from_pretrained(model_id, 
                                                cache_dir=cache_dir,
                                                torch_dtype=torch.bfloat16,
                                                num_labels=26).to(device)
    model.config.pad_token_id = model.config.eos_token_id

    model = PeftModel.from_pretrained(model, experiment_name, is_trainable=False)
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    epoch_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader)):
            if elimination:
                initial_path = [list(val_dataset.classes.keys()), list(val_dataset.classes.keys())] # batch size
                while len(initial_path[0]) > WINDOW:
                    input_ids = []
                    attn_mask = []
                    for index, path in enumerate(initial_path):
                        path = " Available Choices: [" +  (", ".join(path)) + "]"
                        path = val_dataset.to_token(path)
                        input_ids.append(torch.unsqueeze(torch.cat((data['input_ids'][index], torch.tensor(path[0]))),0))
                        attn_mask.append(torch.unsqueeze(torch.cat((data['attention_mask'][index], torch.tensor(path[1]))),0))

                    input_ids = torch.cat(input_ids,0).to(device)
                    attention_mask = torch.cat(attn_mask,0).to(device)
                    labels = data['labels'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                    new_path = []
                    for index, pred in enumerate(outputs[1].data):
                        to_delete = initial_path[index]
                        eliminated = 0
                        top_n = torch.argsort(pred, descending=False)
                        for i in top_n:
                            if class_map[int(i)] in to_delete:
                                to_delete.remove(class_map[int(i)])
                                eliminated += 1
                            if eliminated >= WINDOW:
                                break
                        new_path.append(to_delete)
                    initial_path = copy.deepcopy(new_path)

            else:
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                labels = data['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]

            epoch_loss+= loss
            _, predicted = torch.max(outputs[1].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            # print(predicted, labels)
    acc = 100.0*correct.item()/total
    epoch_loss = epoch_loss.item()
    print("val epoch loss", epoch_loss)
    print("Val Accuracy: ", acc)
