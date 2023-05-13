
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


# [('、', 0.0434032992832596),
#  ('。', 0.02942832788453556),
#  ('▁', 0.026086290201138326),
#  ('で', 0.020166450577257376),
#  ('が', 0.018969096691977688),
#  ('1', 0.018901047813617743),
#  ('の', 0.01702760509325346),
#  ('_', 0.016997254038421998),
#  ('は', 0.016906775921428734),
#  ('2', 0.011865950090033469),
#  ('・', 0.010333400777735759),
#  ('を', 0.008881405837380675),
#  (')', 0.008738751107494362),
#  ('年', 0.008418022539394828),
#  ('に', 0.008288300413060916),
#  ('(', 0.0077313991204202275),
#  ('C', 0.0074696618360155275),
#  ('0', 0.0064392960184500815),
#  ('と', 0.006365692324394565),
#  ('3', 0.005792811392272249),
#  ('START', 0.005656477412719941),
#  ('「', 0.005250730099754063),
#  ('」', 0.005249820999762648),
#  ('ド', 0.004851010048148584),
#  ('ある', 0.004246351179843451),
#  ('ー', 0.004064555042452786),
#  ('月', 0.003910707168052762),
#  ('SE', 0.0038821766992933426),
#  ('TION', 0.003880515981198791),
#  ('ジ', 0.0036824395570852434),
#  ('な', 0.003293540420075731),
#  ('ン', 0.0032081255843994794),
#  ('プ', 0.0030648433130492764),
#  ('4', 0.00301111096998719),
#  ('5', 0.002988280868365535),
#  ('9', 0.0029604232767651154),
#  ('バ', 0.0029023029156342345),
#  ('グ', 0.0028705345238082824),
#  ('や', 0.002820495846853044),
#  ('も', 0.002792819598033064),
#  ('ル', 0.002759352310685099),
#  ('から', 0.0026652640407075602),
#  ('ど', 0.0025956570460893627),
#  ('した', 0.0025048019269999244),
#  ('デ', 0.0024912513263404996),
#  ('6', 0.0024769681962916673),
#  ('し', 0.0024263115222294047),
#  ('00', 0.002402448243977088),
#  ('8', 0.002367787912020968),
#  ('7', 0.0022881855896756476),
#  ('ス', 0.002285200592066044),
#  ('日', 0.0022454125543840734),
#  ('ブ', 0.0022191369399340672),
#  ('だ', 0.0021941760606422477),
#  ('『', 0.0020307504241279545),
#  ('ズ', 0.0019012549762695912),
#  ('』', 0.0018510468869799947),
#  ('パ', 0.0018087057338365725),
#  ('として', 0.001800037071713718),
#  ('LE', 0.0017807574708459328),
#  ('ARTI', 0.0017785861296590907),
#  ('た', 0.0016983777406528277),
#  ('ダ', 0.0016879720055804996),
#  ('する', 0.001614191740922976),
#  ('イ', 0.0015973149318960149),
#  ('ビ', 0.0015669447883508204),
#  ('ま', 0.0015233056026737256),
#  ('て', 0.0015155961484158277),
#  ('ば', 0.0015056366120269496),
#  ('る', 0.0014447412291375005),
#  ('じ', 0.0013668712075632214),
#  ('ア', 0.0013454918481850832),
#  ('ベ', 0.0013435018497786806),
#  ('-', 0.00131328441594333),
#  ('には', 0.001301795396366797),
#  ('ロ', 0.0012870469789207843),
#  ('ラ', 0.0012723486693483144),
#  ('あった', 0.0012235316700718267),
#  ('.', 0.0012072943329548367),
#  ('ィ', 0.0011859937145208368),
#  ('年に', 0.0011674084656149977),
#  ('き', 0.0011672223506561255),
#  ('ヴ', 0.001155513810871692),
#  ('び', 0.0011337598354865931),
#  ('この', 0.0011251102620774692),
#  ('った', 0.0011092833323056845),
#  ('という', 0.0011048571368094293),
#  ('リー', 0.0010938095437379137),
#  ('また', 0.0010909080592508807),
#  ('ガ', 0.0010855966246553744),
#  ('ず', 0.0010798843270715282),
#  ('ん', 0.0010665627909768691),
#  ('ポ', 0.001046708142607953),
#  ('あり', 0.0010422747888440488),
#  ('リ', 0.001026400137287938),
#  ('ピ', 0.0010183566305397567),
#  ('人', 0.001005490837485413),
#  ('ボ', 0.0009885066544437179),
#  ('その', 0.000986676524014808),
#  ('こと', 0.0009583942085338848),
#  ('ュ', 0.0009259338508352027),
#  ('された', 0.0008998037878273911),
#  ('ギ', 0.0008918270915772666),
#  ('第', 0.0008545635362861523),
#  ('ラン', 0.0008536472780270892),
#  ('げ', 0.0008501063216300851),
#  ('か', 0.0008310963488437428),
#  ('れ', 0.0008269135344475513),
#  ('▁(', 0.0008264458609611545),
#  ('ら', 0.0008252814494235952),
#  ('ャ', 0.0008134488329999135),
#  ('ペ', 0.0008036587089453933),
#  ('上', 0.0008023034102705292),
#  ('となった', 0.0007767197616932527),
#  ('ム', 0.000757614345338258),
#  ('して', 0.0007564809529605108),
#  ('日に', 0.0007455121008331333),
#  ('イン', 0.0007356384636560423),
#  ('レ', 0.0007080958358321752),
#  ('している', 0.0006994915981181611),
#  (',', 0.000696936096567493),
#  ('り', 0.0006814646940889386),
#  ('ゴ', 0.0006773224432094243),
#  ('による', 0.000676527875500393),
#  ('大', 0.0006736287771025763),
#  ('され', 0.0006715099298784929),
#  ('ゲ', 0.000670448120177235),
#  ('オ', 0.0006681527023511447),
#  ('によって', 0.0006584747244897911),
#  ('同', 0.0006564012129608176),
#  ('べ', 0.0006462722642375814),
#  ('により', 0.0006429317393347473),
#  ('ている', 0.0006385795126041977),
#  ('S', 0.0006276082743876041),
#  (':', 0.0006208103062103367),
#  ('ため', 0.0006088894044856516),
#  ('ール', 0.0006068182790458944),
#  ('中', 0.0005963862969921869),
#  ('お', 0.0005962932395127508),
#  ('戦', 0.0005792327016161337),
#  ('A', 0.0005782615633050955),
#  ('ク', 0.0005777318514990748),
#  ('ザ', 0.0005691347720527095),
#  ('呼', 0.0005666317644648003),
#  ('い', 0.0005550115099813703),
#  ('ト', 0.0005507905181577178),
#  ('す', 0.0005504349908644876),
#  ('日本', 0.0005476504247490537),
#  ('へ', 0.0005464669244977639),
#  ('への', 0.0005455864575769455),
#  ('▁-', 0.0005395973736440073),
#  ('よ', 0.0005387598563290824),
#  ('ていた', 0.0005337037332797214),
#  ('ョ', 0.0005291653915902996),
#  ('にも', 0.0005170345140145793),
#  ('M', 0.0005165883153311293),
#  ('となる', 0.0005146508108874855),
#  ('ツ', 0.0005102126849451488),
#  ('しかし', 0.0005030663477423002),
#  ('年の', 0.0004889263830464468),
#  ('昭和', 0.00048723464579208286),
#  ('エ', 0.0004871463604910794),
#  ('もの', 0.0004822477193299948),
#  ('シ', 0.00047704365874922256),
#  ('回', 0.0004740157115337251),
#  ('それ', 0.0004734120309619987),
#  ('一', 0.0004725363362196129),
#  ('アル', 0.00047207343491164873),
#  ('していた', 0.0004699164102601044),
#  ('シー', 0.0004655450948158244),
#  ('後', 0.0004602933124507259),
#  ('される', 0.0004587996206013158),
#  ('概要', 0.0004587900762444506),
#  ('月に', 0.00045857294212576637),
#  ('ック', 0.0004524860285349594),
#  ('より', 0.00045055329626974823),
#  ('け', 0.00045054136582366666),
#  ('リン', 0.000449004724368363),


# import os
# from os.path import join
# import urllib
# import gzip
# from tqdm import tqdm


# url_file = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-43/wet.paths.gz"


# index_name = url_file.split("/")[-2]
# urllib.request.urlretrieve(url_file, index_name)

# dir_name = index_name + "_FILES"
# os.mkdir(dir_name)


# base_url = "https://data.commoncrawl.org/"

# # read urls
# with gzip.open('wet.paths.gz', 'rb') as r:
#     cc_urls = [base_url + line.decode().strip() for line in r.readlines()]
#     cc_names = [line.split("/")[-1] for line in cc_urls]

# cc_urls[0]


# urllib.request.urlretrieve(cc_urls[0], join(dir_name, cc_names[0]))


# # read file
# with gzip.open(join(dir_name, cc_names[0]), 'rb') as r:
#     text = r.read().decode()
#     docs = text.split("WARC/1.0")
    
    
# import re
# re_hira_kata = re.compile("[ぁ-んァ-ヴー]")
# re_kan = re.compile("[一-龠]")
# re_ten = re.compile("[、。]")


# print("source", len(docs))
# docs = [re.split("Content\-Length:.+\r\n\r\n", doc)[-1] for doc in docs]
# docs = [doc for doc in tqdm(docs) if len(doc)>500 and detect_ja(doc)]
# print("ja", len(docs))


# def filter_text(text):
#     lines = text.replace("\r", "\n")
#     lines = text.split("\n")
#     filter_texts = []
#     for line in lines:
#         if line.strip().endswith("。"): filter_texts.append(line)
#         if line.strip().endswith("？"): filter_texts.append(line)
#         if line.strip().endswith("」"): filter_texts.append(line)
#         # if line.strip().endswith("）"): filter_texts.append(line)
#     return "\n".join(filter_texts)


# docs = [filter_text(doc) for doc in docs]
# docs = [doc for doc in docs if len(doc) > 50]




# import threading
# import requests

# urls = [
#     f"https://example.com/file{i}.zip" for i in range(1, 1001)
# ]

# class DownloadThread(threading.Thread):
#     def __init__(self, urls):
#         threading.Thread.__init__(self)
#         self.urls = urls

#     def run(self):
#         for url in self.urls:
#             response = requests.get(url)
#             with open(url.split("/")[-1], "wb") as f:
#                 f.write(response.content)

# def download_in_parallel(urls, num_threads):
#     threads = []
#     for i in range(0, len(urls), num_threads):
#         thread = DownloadThread(urls[i:i+num_threads])
#         thread.start()
#         threads.append(thread)

#     for thread in threads:
#         thread.join()

# download_in_parallel(urls, 10)



# import datasets
# from torch.utils.data import Dataset

# class QADataset(Dataset):
#     def __init__(self, tokenizer, max_length, mode="train"):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.data = [self.preprocess(data) for data in datasets.load_dataset("shunk031/JGLUE", "JCommonsenseQA")[mode]]
      
#     def preprocess(self, data):
#         option = [data["choice0"], data["choice1"], data["choice2"], data["choice3"], data["choice4"]]
#         context = "質問：" + data["question"] + "\n選択肢：" + "、".join(option) + "\n答え："
#         answer = option[data["label"]]
#         return context + answer + "</s>"

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         tokenized = self.tokenizer(
#             self.data[index], 
#             max_length=self.max_length, 
#             padding="max_length", 
#             truncation=True, 
#             return_tensors="pt", 
#             add_special_tokens=False
#             )
        
#         tokenized["labels"] = tokenized["input_ids"].clone()
#         tokenized["labels"][tokenized["labels"]==tokenizer.pad_token_id] = -100
#         return tokenized


# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token_id = tokenizer.eos_token_id

# QADataset(tokenizer, 512)

# import torch
# from torch.utils.data import DataLoader
# device = torch.device("cuda")
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm

# tokenizer = AutoTokenizer.from_pretrained('rinna/japanese-gpt2-small', use_fast=False)
# model = AutoModelForCausalLM.from_pretrained('rinna/japanese-gpt2-small').to(device)

# # Add PAD Token
# tokenizer.pad_token_id = tokenizer.eos_token_id
# model.config.pad_token_id = tokenizer.eos_token_id

# batch_size = 4
# num_epochs = 2
# learning_rate = 5e-5

# dataset = QADataset(tokenizer, max_length=512)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# model.train()
# for epoch in range(num_epochs):
#     total_loss = 0
#     for batch in tqdm(dataloader):
#         for key in batch: batch[key] = batch[key].to(device).squeeze()

#         optimizer.zero_grad()
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
        
#     avg_loss = total_loss / len(dataloader)
#     print(f'Epoch {epoch + 1} loss: {avg_loss}')

#     # Save the model
#     model.save_pretrained('qa_model')
#     tokenizer.save_pretrained('qa_model')
#     print("save!")

# class QAGenDataset(Dataset):
#     def __init__(self, tokenizer, max_length, mode="validation"):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.raw_data = datasets.load_dataset("shunk031/JGLUE", "JCommonsenseQA")[mode]
#         self.data = [self.preprocess(data) for data in self.raw_data]
      
#     def preprocess(self, data):
#         option = [data["choice0"], data["choice1"], data["choice2"], data["choice3"], data["choice4"]]
#         context = "質問：" + data["question"] + "\n選択肢：" + "、".join(option) + "\n答え："
#         answer = option[data["label"]]
#         return context, answer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         context, answer = self.data[index]
#         tokenized = self.tokenizer(
#             context, 
#             max_length=self.max_length, 
#             # padding="max_length", 
#             truncation=True, 
#             return_tensors="pt", 
#             add_special_tokens=False
#             )
#         return tokenized, answer

# import torch
# from torch.utils.data import DataLoader
# device = torch.device("cuda")
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained('qa_model', use_fast=False)
# model = AutoModelForCausalLM.from_pretrained('qa_model').to(device)

# test_dataset = QAGenDataset(tokenizer, max_length=5000)

# for context, answer in test_dataset:
#     for key in context: context[key] = context[key].to(device)
#     input_length = context["input_ids"].size(-1)
#     gen_ids = model.generate(**context, max_length=input_length+20, do_sample=False)
#     output = tokenizer.batch_decode(gen_ids.cpu(), skip_special_tokens=False)
#     predict = output.split("</s>")[0]
#     print(predict, answer, predict.strip()==answer.strip())


# import pandas as pd
# import datasets
# from torch.utils.data import Dataset

# class QADataset(Dataset):
#     def __init__(self, tokenizer, max_length, mode="train"):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.data = [self.preprocess(data) for data in datasets.load_dataset("shunk031/JGLUE", "JCommonsenseQA")[mode]]
      
#     def preprocess(self, data):
#         option = [data["choice0"], data["choice1"], data["choice2"], data["choice3"], data["choice4"]]
#         context = "質問：" + data["question"] + "\n選択肢：" + "、".join(option) + "\n答え："
#         answer = option[data["label"]]
#         return context, answer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         context, answer = self.data[index]
#         tokenized_context = self.tokenizer(
#             context, 
#             max_length=self.max_length, 
#             # padding="max_length", 
#             truncation=True, 
#             add_special_tokens=False
#             )
        
#         tokenized_answer = self.tokenizer(
#             answer, 
#             max_length=self.max_length, 
#             # padding="max_length", 
#             truncation=True, 
#             add_special_tokens=False
#             )

        
#         tokenized = {}
#         tokenized["input_ids"] = tokenized_context["input_ids"] + tokenized_answer["input_ids"] + [2]
#         tokenized["attention_mask"] = tokenized_context["attention_mask"] + tokenized_answer["attention_mask"] + [0]
#         tokenized["labels"] = [-100]*len(tokenized_context["input_ids"]) + tokenized_answer["input_ids"] + [2]

#         # print(tokenized)

#         if len(tokenized["input_ids"]) < 512: tokenized["input_ids"] += (512-len(tokenized["input_ids"]))*[2]
#         if len(tokenized["attention_mask"]) < 512: tokenized["attention_mask"] += (512-len(tokenized["labels"]))*[0]
#         if len(tokenized["labels"]) < 512: tokenized["labels"] += (512-len(tokenized["labels"]))*[-100]

#         if len(tokenized["input_ids"]) > 512: tokenized["input_ids"] = tokenized["input_ids"][:512]
#         if len(tokenized["attention_mask"]) > 512: tokenized["attention_mask"] = tokenized["attention_mask"][:512]
#         if len(tokenized["labels"]) > 512: tokenized["labels"] = tokenized["labels"][:512]

#         for key in tokenized: tokenized[key] = torch.tensor(tokenized[key])

#         # print(tokenized)

#         return tokenized


# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small")
# tokenizer.pad_token_id = tokenizer.eos_token_id

# QADataset(tokenizer, 512)[0]


########
# import json
# import datasets
# import random
# import torch

# INSTRUCTION = """[問題]から[質問]の回答を抜き出してください。\n"""
# PROMPT_TEMPLETE = """[問題]:{CONTEXT}\n[質問]:{QUESTION}\n[回答]:{ANSWER}。"""

# class FewDataset(torch.utils.data.Dataset):
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#         self.squad = datasets.load_dataset("shunk031/JGLUE", "JSQuAD")
#         self.train = self.squad["train"]
#         self.val = self.squad["validation"]
#         self.incontext_prompt = self.make_incontext_prompt(random.sample(list(self.train), 2))
#         print(self.incontext_prompt)
      
#     def make_incontext_prompt(self, samples):
#         prompts = [self.make_prompt_sample(sample) for sample in samples]
#         return "\n".join(prompts)
    
#     def make_prompt_sample(self, sample, prefix=False):
#         question = sample["question"]
#         context = sample["context"]
#         if not prefix:
#             answer = sample["answers"]["text"][0]

#         prompt = PROMPT_TEMPLETE.replace("{QUESTION}", question)
#         prompt = prompt.replace("{QUESTION}", question)
#         prompt = prompt.replace("{CONTEXT}", context)

#         if not prefix:
#             prompt = prompt.replace("{ANSWER}", answer)
#         else:
#             prompt = prompt.replace("{ANSWER}。", "")
#         return prompt

#     def __len__(self):
#         return len(self.val)

#     def __getitem__(self, idx):
#         prefix_sample = self.val[idx]
#         prompt = self.incontext_prompt + "\n" + self.make_prompt_sample(prefix_sample, True)
#         print(prompt)
#         return self.tokenizer(prompt, 
#                               return_tensors="pt", 
#                               add_special_tokens=False,
#                               max_length=512,
#                               truncation=True)


# !pip install slack_bolt
# !pip install slack_sdk

# # update message

# from slack_sdk import WebClient
# from slack_sdk.errors import SlackApiError
# from slack_bolt import App
# from slack_bolt.adapter.socket_mode import SocketModeHandler

# TOKEN = "xoxb-"
# APP_TOKEN = "xapp-1-"

# # Initialize a Slack app
# app = App(token=TOKEN)
# client = WebClient(token=TOKEN)

# # Define a handler function for receiving messages
# @app.event("message")
# def handle_mention(event, say):
#     # Get the text of the message that mentioned the bot
#     message_text = event["text"]
#     update_message = "START:\n"

#     try:
#         # Post a reply message to the original message's thread
#         response = client.chat_postMessage(
#             channel=event["channel"],
#             thread_ts=event["ts"],
#             text=update_message
#         )
#     except SlackApiError as e:
#         print(f"Error posting message: {e}")

#     print(response)
    
#     for i in range(5):
#         update_message += "ごんごん"
#         import time; time.sleep(0.5)
#         # Send a reply message
#         try:
#             # Post a reply message to the original message's thread
#             response = client.chat_update(
#                 channel=response["channel"],
#                 ts=response["ts"],
#                 text=update_message
#             )
#         except SlackApiError as e:
#             print(f"Error posting message: {e}")


# # Start the Socket Mode handler
# handler = SocketModeHandler(app=app, app_token=APP_TOKEN)
# handler.start()
