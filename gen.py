
import random
import transformers
from transformers import AutoTokenizer
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from gpt_simple import GPT1, GPT1Config

tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
PAD_IDX = tokenizer.pad_token_id
BOS_IDX = tokenizer.cls_token_id
EOS_IDX = tokenizer.sep_token_id
UNK_IDX = tokenizer.unk_token_id
VOCAB_SIZE = tokenizer.vocab_size

model = GPT1(GPT1Config(vocab_size=VOCAB_SIZE))
model.load_state_dict(torch.load("model.bin"))

device = torch.device("cuda")
model = model.to(device)

def top_k(logits, thres = 0.9, k = None):
    if k == None:
        k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

#可変長Generate
text = "猫は"
model.eval()
max_seq_len = 128
seq_len = 128
if text == None:
    start_tokens = torch.tensor([BOS_IDX])[None,:].to(device)
else:
    start_tokens = torch.tensor(tokenizer(text)["input_ids"])[None,:-1].to(device)
b, t = start_tokens.shape
out = start_tokens

SAMPLING = True

for _ in range(seq_len):
    x = out[:, -max_seq_len:]
    with torch.no_grad():
        logits = model(x)[:, -1, :]
    if SAMPLING:
        temperature = 1.
        filtered_logits = top_k(logits, thres = None, k=10)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        probs[:, UNK_IDX] = 0.0
        pred = torch.multinomial(probs, 1)
    else:
        pred = logits.argmax().unsqueeze(0).unsqueeze(0)
    out = torch.cat((out, pred), dim=-1)
    if pred.item() == EOS_IDX:
        break
out = out.squeeze(0)
print(tokenizer.decode(out.tolist()).replace(" ",""))
