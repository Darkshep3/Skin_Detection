from peft import LoraConfig, get_peft_model, TaskType 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataset import SkinDiseaseDataset, WINDOW
import torch
import torch.nn as nn
from tqdm import tqdm
import copy

batch_size = 2
learning_rate = 1e-4
num_epochs = 50

LORA_R = 16 
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

lora_config = LoraConfig (
    r = LORA_R,
    lora_alpha = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT, 
    task_type = TaskType.SEQ_CLS,
    bias = "none",
)
experiment_name = 'llama-lora-pred1-coo'

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# model_id = "mistralai/Mistral-7B-v0.1"
# model_id = "tiiuae/falcon-7b"
model_id = "D:\Allen_2023\model_weights\llama-7b"
cache_dir="D:\Allen_2023\model_weights"
path = "D:\Allen_2023\LLM\split_descriptions_v4.pickle"

criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    # mistral and llama don't have default pad token id
    # https://github.com/huggingface/blog/blob/main/Lora-for-sequence-classification-with-Roberta-Llama-Mistral.md
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = SkinDiseaseDataset(path, tokenizer, split="train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                    shuffle=True, pin_memory=True, drop_last=True)
    val_dataset = SkinDiseaseDataset(path, tokenizer, split="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                    pin_memory=True)
    
    class_map = dict((v,k) for k,v in val_dataset.classes.items())

    print(len(train_dataset), len(val_dataset))

    model = AutoModelForSequenceClassification.from_pretrained(model_id, 
                                                cache_dir=cache_dir,
                                                torch_dtype=torch.bfloat16,
                                                num_labels=26).to(device)
    
    model.config.pad_token_id = model.config.eos_token_id

    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()

    model.train()

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
        for i, data in enumerate(train_loader):

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+= loss
            _, predicted = torch.max(outputs[1].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
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
            for i, data in enumerate(val_loader):

                initial_path = [list(val_dataset.classes.keys()), list(val_dataset.classes.keys())] # batch size
                while len(initial_path[0]) > WINDOW:
                    input_ids = []
                    attn_mask = []
                    for index, path in enumerate(initial_path):
                        path = " Top Choices: [" +  (", ".join(path)) + "]"
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

                loss = outputs[0]

                epoch_loss+= loss
                _, predicted = torch.max(outputs[1].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                # print(predicted, labels)
        acc = 100.0*correct.item()/total
        val_acc.append(acc)   
        epoch_loss = epoch_loss.item()
        val_loss.append(epoch_loss)
        print("val epoch loss", epoch, epoch_loss)
        print("Val Accuracy: ", acc)

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(experiment_name)
