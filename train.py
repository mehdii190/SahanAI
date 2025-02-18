
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

from tqdm import tqdm
import numpy as np
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


# --- توابع کمکی ---
def simple_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def acc_and_f1(y_true, y_pred, average='weighted'):
    return {
        "acc": simple_accuracy(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average)
    }

def y_loss(y_true, y_pred, losses):
    y_true = torch.stack(y_true).cpu().detach().numpy()
    y_pred = torch.stack(y_pred).cpu().detach().numpy()
    return [y_true, y_pred], np.mean(losses)

# --- آموزش و ارزیابی ---
def eval_op(model, data_loader, loss_fn):
    model.eval()
    losses, y_pred, y_true = [], [], []
    
    with torch.no_grad():
        for dl in tqdm(data_loader, desc="Evaluation"):
            inputs = {k: v.to(device) for k, v in dl.items() if k != 'targets'}
            targets = dl['targets'].to(device)
            
            outputs = model(**inputs)
            loss = loss_fn(outputs, targets)
            
            _, preds = torch.max(outputs, dim=1)
            losses.append(loss.item())
            y_pred.extend(preds)
            y_true.extend(targets)
    
    return y_loss(y_true, y_pred, losses)

def train_op(model, data_loader, loss_fn, optimizer, scheduler, 
            eval_data_loader, print_every_step=100, clip=0.0):
    
    model.train()
    step = 0
    history = collections.defaultdict(list)
    eval_loss_min = [np.Inf]  # استفاده از لیست به جای nonlocal

    # --- تابع callback اصلاح شده ---
    def eval_cb(train_score, train_loss, eval_score, eval_loss):
        statement = (
            f"Step: {step} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_score['acc']:.3f} | "
            f"Valid Loss: {eval_loss:.4f} | "
            f"Valid Acc: {eval_score['acc']:.3f}"
        )
        print(statement)

        if eval_loss <= eval_loss_min[0]:
            print(f"🔄 Validation loss improved ({eval_loss_min[0]:.4f} → {eval_loss:.4f})")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': config.__dict__,
                'epoch': epoch
            }, 'model_save/model_sahan.pt')
            eval_loss_min[0] = eval_loss

    # --- حلقه آموزش ---
    for epoch in range(1, EPOCHS + 1):
        losses, y_pred, y_true = [], [], []
        
        for dl in tqdm(data_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            step += 1

            input_ids = dl['input_ids']
            attention_mask = dl['attention_mask']
            token_type_ids = dl['token_type_ids']
            targets = dl['targets']

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
            loss.backward()
            
            
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            scheduler.step()

            _, preds = torch.max(outputs, dim=1)
            losses.append(loss.item())
            y_pred.extend(preds)
            y_true.extend(targets)


        train_y, train_loss = y_loss(y_true, y_pred, losses)
        train_score = acc_and_f1(train_y[0], train_y[1])
        
        eval_y, eval_loss = eval_op(model, eval_data_loader, loss_fn)
        eval_score = acc_and_f1(eval_y[0], eval_y[1])
        
        eval_cb(train_score, train_loss, eval_score, eval_loss)
        
        history['train_acc'].append(train_score['acc'])
        history['train_loss'].append(train_loss)
        history['val_acc'].append(eval_score['acc'])
        history['val_loss'].append(eval_loss)
    
    return history




