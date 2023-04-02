
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

optim = torch.optim.Adam(model.parameters(), lr=1e-4)
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

    
    
    
    
    
# MEMO
    
# import re
# import time
# import torch
# import transformers
# from more_itertools import chunked
# from parsers import splitter_ja, splitter_en

# from transformers import pipeline
# tokenizer_en_ja = transformers.AutoTokenizer.from_pretrained("staka/fugumt-en-ja")
# tokenizer_ja_en = transformers.AutoTokenizer.from_pretrained("staka/fugumt-ja-en")
# fugu_translator_en_ja = pipeline('translation', model='staka/fugumt-en-ja', device=0)
# fugu_translator_ja_en = pipeline('translation', model='staka/fugumt-ja-en', device=0)

# def en_ja(text):
#     now = time.time()
#     translated_text = fugu_translator_en_ja(text)[0]["translation_text"]
#     if translated_text == "「この版権表示を残す」んだから、「禁無断複製」とかいうのはダメだぞ":
#         translated_text = None
#     # print("en_ja time:", time.time()-now)
#     return translated_text

# def ja_en(text):
#     now = time.time()
#     translated_text = fugu_translator_ja_en(text)[0]["translation_text"]
#     translated_text = re.sub("[□■◇◆]", "", translated_text)
#     # print("ja_en time:", time.time()-now)
#     return translated_text

# def chunker(text, trans_en_ja, max_length=512):

#     tokenizer = tokenizer_en_ja if trans_en_ja else tokenizer_ja_en
#     splitter = splitter_en if trans_en_ja else splitter_ja

#     sentences = splitter(text)
#     chunks = []
#     chunk = [""]
#     for sentence in sentences:
#         if len(tokenizer.tokenize(" ".join(chunk))) > max_length:
#             # print(len(tokenizer.tokenize(" ".join(chunk))))
#             chunks.append(" ".join(chunk))
#             chunk = [""]
#         chunk.append(sentence)
#     chunks.append(" ".join(chunk))

#     chunks = [chunk for chunk in chunks if len(tokenizer.tokenize(chunk)) <= 500]
#     return chunks



# def translator_to_chunks(text, trans_en_ja=True, chunk_max_length=512, batch=1):
#     chunks = chunker(text, trans_en_ja, chunk_max_length)
#     # batchs = chunked(chunks)
#     translator = en_ja if trans_en_ja else ja_en
#     results = []
#     for chunk in chunks:
#         results.append(translator(chunk))
#     return chunks, results

# def translator_chunks(chunks, trans_en_ja=True):
#     results = []
#     tokenizer = tokenizer_en_ja if trans_en_ja else tokenizer_ja_en
#     translator = en_ja if trans_en_ja else ja_en
#     for chunk in chunks:
#         if len(tokenizer.tokenize(chunk)) > 512 or chunk is None:
#             results.append(None)
#         else:
#             results.append(translator(chunk))
#     return chunks, results




# import json
# import re


# with open("train.txt") as r:
#     lines = r.readlines()

# # print(json.loads(lines[0]))

# re_katakana = re.compile('[a-zA-Z=\-\u30A1-\u30F4・＝]+')
# re_title = re.compile("_START_ARTICLE_\n.+\n")
# def en_detect(line):
#     text = json.loads(line)["data"]
#     title = re_title.findall(text)[0]
#     # return re_katakana.fullmatch(title)
#     return bool(re_katakana.findall(title.strip().replace("_START_ARTICLE_", "")))


# lines = [line.strip() for line in lines if en_detect(line)]
# with open("train_en_page.txt", "w") as w:
#     w.write("\n".join(lines))



        
# import functools

# from ja_sentence_segmenter.common.pipeline import make_pipeline
# from ja_sentence_segmenter.concatenate.simple_concatenator import  concatenate_matching
# from ja_sentence_segmenter.normalize.neologd_normalizer import  normalize
# from ja_sentence_segmenter.split.simple_splitter import  split_newline, split_punctuation

# split_punc2 = functools.partial(split_punctuation, punctuations=r".。 !?")
# concat_tail_no = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(の)$", remove_former_matched=False)
# concat_decimal = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(\d.)$", latter_matching_rule=r"^(\d)(?P<result>.+)$", remove_former_matched=False, remove_latter_matched=False)
# segmenter = make_pipeline(normalize, split_newline, concat_tail_no, split_punc2)
# segmenter = make_pipeline(normalize, split_newline, concat_tail_no, split_punc2, concat_decimal)

# def splitter_ja(text):
#     return list(segmenter(text))

# def splitter_en(text):
#     return "NOT IMPLEMENT"

