from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from database.engine import Base, engine, get_db
from database.models import Analysis
from config import Settings
from model import SentimentModel
from predict import predict_sentiment_farsi
from fastapi.middleware.cors import CORSMiddleware

import os
import re
import json
import copy
import collections

import torch

from transformers import AutoTokenizer, BertConfig



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = BertConfig.from_pretrained(Settings.MODEL_NAME)
pt_model = SentimentModel(config)
pt_model = torch.load(Settings.MODEL_SAVE2, map_location=device)


path = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
tokenizer = AutoTokenizer.from_pretrained(path)


app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def init_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/process-text/", response_class=HTMLResponse)
async def process_text(request: Request, text: str = Form(...), db: AsyncSession = Depends(get_db)):
    
    
    predicted_class, probabilities = predict_sentiment_farsi(pt_model, tokenizer, text, device)
    
    
    probabilities_list = probabilities.cpu().numpy().tolist()
    probabilities_json = json.dumps(probabilities_list)
    
    label_mapping = {0: "negetive", 1: "positive"}
    new_record = Analysis(
        text=text,
        result=label_mapping[predicted_class],
        accuracy=probabilities_json
    )
    
    async with db.begin():
        db.add(new_record)
        await db.flush()
        
    return templates.TemplateResponse("index.html", 
                                      
    {
        "request": request,
        "result": label_mapping[predicted_class],
        "accuracy": probabilities
    })



