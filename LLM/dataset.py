from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
import pickle
import json
import random
import copy
from itertools import combinations

MAX_LEN = 512
CHOICES = 3
PROB = 0.92

class SkinDiseaseDataset(Dataset):
    def __init__(self, path, tokenizer, split="train"):
        self.encodings = []
        self.labels = []
        self.classes = {}
        self.tokenizer = tokenizer

        texts = []
        targets = []
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
            texts = data[split]
            targets = data[split+"_labels"]
            self.classes = data['classes']

        self.labels = [self.classes[x] for x in targets]

        self.options_map = "Options map: " + json.dumps(self.classes)
        self.options_map = self.options_map.replace('"', '')
        self.options_map = self.tokenizer(self.options_map)

        # generate all 26C3 possibilities and create mapping of which ones contain which class
        self.predictions = []
        combos, combo_map = self.all_options()
        self.predictions = self.tokenizer(combos, padding=True)
        self.predictions_map = combo_map
        self.combos_len = len(combos)
        self.predictions_per = len(list(combo_map.values())[0])
            
        self.encodings = self.tokenizer(texts, truncation=True, max_length = MAX_LEN, padding=True)

    def __getitem__(self, index):
        item = {}

        for key, val in self.encodings.items():
            item[key] = copy.deepcopy(val[index])

        # add mapping to prompt 
        for key, val in item.items():
            if key == 'input_ids':
                item['input_ids'] += self.options_map['input_ids']
            elif key == 'attention_mask':
                item['attention_mask'] += self.options_map['attention_mask']

        # add chain of elimination 

        # add most likely one, three, five predictions
        # at runtime select correct one with probability 80%
        label = self.labels[index]
        key_val = ""
        key_attn = ""
        if random.uniform(0, 1) > PROB: # pick incorrect, else pick correct
            random_selection = random.choice([i for i in range(self.combos_len) if i not in self.predictions_map[label]])
            key_val = self.predictions['input_ids'][random_selection]
            key_attn = self.predictions['attention_mask'][random_selection]
        else:
            random_selection = random.randint(0, self.predictions_per-1)
            random_selection = self.predictions_map[label][random_selection]
            key_val = self.predictions['input_ids'][random_selection]
            key_attn = self.predictions['attention_mask'][random_selection]
        for key, val in item.items():
            if key == 'input_ids':
                item['input_ids'] += key_val
            elif key == 'attention_mask':
                item['attention_mask'] += key_attn

        item = {key: torch.tensor(val) for key, val in item.items()}
    

        # original
        # item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        
        item['labels'] = torch.tensor(self.labels[index])
        return item
    
    def all_options(self):
        combos = combinations(self.classes.keys(), CHOICES)
        combo_map = {self.classes[i]:[] for i in self.classes.keys()}
        combos = list(combos)
        for i, x in enumerate(combos):
            for c in x:
                combo_map[self.classes[c]].append(i)
        combo_text = [" " for _ in range(len(combos))]
        for i, x in enumerate(combos):
            combo_text[i] = "Top Choices: " +  (", ".join(x))

        return combo_text, combo_map

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    path = "D:\Allen_2023\LLM\split_descriptions_more.pickle"

    model_id = "D:\Allen_2023\model_weights\llama-7b"
    cache_dir="D:\Allen_2023\model_weights"

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    # mistral and llama don't have default pad token id
    # https://github.com/huggingface/blog/blob/main/Lora-for-sequence-classification-with-Roberta-Llama-Mistral.md
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = SkinDiseaseDataset(path, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                    shuffle=True, pin_memory=True, drop_last=True)

    for i, data in enumerate(dataloader):
        print(i, data, data['input_ids'].shape)
        print(tokenizer.batch_decode(data['input_ids']))
