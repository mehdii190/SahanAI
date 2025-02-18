from transformers import AutoTokenizer, BertConfig
import torch
import numpy as np
from config import Settings
from data_loader import DataLoader
from model import SentimentModel

import os
import re
import json
import copy
import collections

config = BertConfig.from_pretrained(Settings.MODEL_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


path = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
tokenizer = AutoTokenizer.from_pretrained(path)

def predict_sentiment_farsi(pt_model, tokenizer, text, device, max_len=128):
    
    pt_model = SentimentModel(config)
    state_dict = torch.load(Settings.MODEL_SAVE2, map_location=device)
    pt_model.load_state_dict(state_dict)
    pt_model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        return_token_type_ids=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)

    with torch.no_grad():
        logits = pt_model(input_ids, attention_mask, token_type_ids)

    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities



