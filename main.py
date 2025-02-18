from transformers import BertConfig, BertTokenizer, BertModel, AutoTokenizer, BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from config import Settings
from data_loader import DataLoader, create_data_loader
from model import SentimentModel
from train import train_op, acc_and_f1
from config import Settings
from predict import predict_sentiment_farsi


import os
import re
import json
import copy
import collections


MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

EPOCHS = 3
EEVERY_EPOCH = 1000
LEARNING_RATE = 2e-5
CLIP = 0.0


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


path = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"

tokenizer = BertTokenizer.from_pretrained(Settings.MODEL_NAME)
config = BertConfig.from_pretrained(Settings.MODEL_NAME)

pt_model = SentimentModel(config=config)
pt_model = pt_model.to(device)

Loader = DataLoader()
train, valid, test= Loader.load()
label_list = [0, 1]
train_data_loader = create_data_loader(train['comment'].to_numpy(), train['label'].to_numpy(), tokenizer, MAX_LEN, TRAIN_BATCH_SIZE, label_list)
valid_data_loader = create_data_loader(valid['comment'].to_numpy(), valid['label'].to_numpy(), tokenizer, MAX_LEN, VALID_BATCH_SIZE, label_list)
test_data_loader = create_data_loader(test['comment'].to_numpy(), None, tokenizer, MAX_LEN, TEST_BATCH_SIZE, label_list)



optimizer = AdamW(pt_model.parameters(), lr=LEARNING_RATE, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss()

step = 0
eval_loss_min = np.Inf
history = collections.defaultdict(list)


def Train():
    
    history = train_op(
    model=pt_model,
    data_loader=train_data_loader,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    scheduler=scheduler,
    eval_data_loader=valid_data_loader,
    clip=1.0
)



pt_model = torch.load(Settings.MODEL_SAVE, map_location=device)
pt_model = pt_model.to(device)

path = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
tokenizer = AutoTokenizer.from_pretrained(path)

def pred():
    
    text = input("enter your type for predict: ")
    
    predicted_class, probabilities = predict_sentiment_farsi(pt_model, tokenizer, text, device)
    
    label_mapping = {0: "negetive", 1: "positive"}
    print(f"text: {text}")
    print(f"class: {label_mapping[predicted_class]}")
    print(f"probably: {probabilities}")


while True:
    
    types = input("enter your type please (1 : predict , 2 : Train your Data ): ")
    
    if types == "1":
        pred()
        
    elif types == "2":
        Train()
        
    elif types == "exit":
        break
    
    else:
        print("wrong")





