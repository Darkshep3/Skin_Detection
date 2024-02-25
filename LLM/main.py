from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, TaskType 
from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer
from dataset import SkinDiseaseDataset
import torch
import torch.nn as nn
from tqdm import tqdm

batch_size = 2
learning_rate = 1e-4
num_epochs = 20

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

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
model_id = "mistralai/Mistral-7B-v0.1"
cache_dir="D:\Allen_2023\model_weights"
path = "D:\Allen_2023\LLM\split_descriptions_v3.pickle"

criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    # mistral and llama don't have default pad token id
    # https://github.com/huggingface/blog/blob/main/Lora-for-sequence-classification-with-Roberta-Llama-Mistral.md
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = SkinDiseaseDataset(path, tokenizer, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                    shuffle=True, pin_memory=True, drop_last=True)
    val_dataset = SkinDiseaseDataset(path, tokenizer, "val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                    pin_memory=True)

    print(len(train_dataset), len(val_dataset))
    # bfloat16 for mistral
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
        val_acc.append(acc)   
        epoch_loss = epoch_loss.item()
        val_loss.append(epoch_loss)
        print("val epoch loss", epoch, epoch_loss)
        print("Val Accuracy: ", acc)
