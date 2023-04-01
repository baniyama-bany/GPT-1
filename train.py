
import random
import transformers
from transformers import AutoTokenizer
import torch
torch.manual_seed(41)
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from gpt_simple import GPT1, GPT1Config
with open("text.txt","r",encoding="utf-8") as r:
    lines  = [line.strip() for line in r.readlines()]

tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
PAD_IDX = tokenizer.pad_token_id
BOS_IDX = tokenizer.cls_token_id
EOS_IDX = tokenizer.sep_token_id
VOCAB_SIZE = tokenizer.vocab_size
TRAIN_BATCH = 8
VAL_BATCH = 8

model = GPT1(GPT1Config(vocab_size=VOCAB_SIZE))

class MyDataset(Dataset):
    def __init__(self,lines):
        self.text = lines

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        encode = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        input_ids = encode["input_ids"].squeeze()
        labels = input_ids.clone()
        labels[labels==tokenizer.pad_token_id] = -100
        return input_ids, labels

dataset = MyDataset(lines)

train_length = int(len(dataset)*0.9)
val_length = len(dataset) - train_length

train,val = torch.utils.data.random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(40))

train_loader = DataLoader(train, batch_size=TRAIN_BATCH, shuffle=True)
val_loader = DataLoader(val, batch_size=VAL_BATCH, shuffle=False)

optim = torch.optim.Adam(model.parameters(), lr=1e-5)
device = torch.device("cuda")
model = model.to(device)

from tqdm import tqdm
epoch = 100
for i in tqdm(range(epoch)):
    model.train()
    step = 0
    train_epoch_loss = 0
    for batch in (train_loader):
        step += 1
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optim.zero_grad()
        output = model(input_ids[:,:-1])
        loss = F.cross_entropy(output.reshape(-1, output.size(-1)), labels[:,1:].reshape(-1), ignore_index = -100)
        loss.backward()
        optim.step()
        train_epoch_loss += loss.item()
    train_epoch_loss /= step

    model.eval()
    step = 0
    val_epoch_loss = 0
    for batch in val_loader:
        step += 1
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(input_ids[:,:-1])
        loss = F.cross_entropy(output.reshape(-1, output.size(-1)), labels[:,1:].reshape(-1), ignore_index = -100)
        val_epoch_loss += loss.item()
    val_epoch_loss /= step

    torch.save(model.state_dict(), "model.bin")

    print("\EPOCH:{}\tTRAINLOSS:{}\tVALLOSS:{}".format(i, train_epoch_loss, val_epoch_loss))