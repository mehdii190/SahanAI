import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import Settings
from preprocess import TextPreprocessor
import torch


from transformers import BertConfig, BertTokenizer, BertModel, AutoTokenizer, BertForSequenceClassification



minlim, maxlim = 3, 256


MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

EPOCHS = 3
EEVERY_EPOCH = 1000
LEARNING_RATE = 2e-5
CLIP = 0.0



class DataLoader:
    def __init__(self):
        self.settings = Settings()
        self.preprocessor = TextPreprocessor()

    def load(self):
        df = pd.read_csv(self.settings.DATA_PATH)
        if "cleaned_comment" not in df.columns:
            df["cleaned_comment"] = df["comment"].apply(self.preprocessor.clean)
        else:
            pass
        
        data = df[['cleaned_comment', 'label']]
        data.columns = ['comment', 'label']
        
        
        negative_data = data[data['label'] == 0]
        positive_data = data[data['label'] == 1]

        cutting_point = min(len(negative_data), len(positive_data))

        if cutting_point <= len(negative_data):
            negative_data = negative_data.sample(n=cutting_point).reset_index(drop=True)

        if cutting_point <= len(positive_data):
            positive_data = positive_data.sample(n=cutting_point).reset_index(drop=True)

        new_data = pd.concat([negative_data, positive_data])
        new_data = new_data.sample(frac=1).reset_index(drop=True)
        
        labels = list(sorted(data['label'].unique()))
        
        
        new_data['label_id'] = new_data['label'].apply(lambda t: labels.index(t))

        train, test = train_test_split(new_data, test_size=0.1, random_state=1, stratify=new_data['label'])
        train, valid = train_test_split(train, test_size=0.1, random_state=1, stratify=train['label'])

        train = train.reset_index(drop=True)
        valid = valid.reset_index(drop=True)
        test = test.reset_index(drop=True)
         
        return train, valid, test
    
    
    
    
class SahanDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Taaghche. """

    def __init__(self, tokenizer, comments, targets=None, label_list=None, max_len=128):
        self.comments = comments
        self.targets = targets
        self.has_target = isinstance(targets, list) or isinstance(targets, np.ndarray)

        self.tokenizer = tokenizer
        self.max_len = max_len


        self.label_map = {label: i for i, label in enumerate(label_list)} if isinstance(label_list, list) else {}

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])

        if self.has_target:
            target = self.label_map.get(str(self.targets[item]), self.targets[item])
            target = int(target) 

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt')

        inputs = {
            'comment': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }

        if self.has_target:
            inputs['targets'] = torch.tensor(target, dtype=torch.long)

        return inputs


def create_data_loader(x, y, tokenizer, max_len, batch_size, label_list):
    dataset = SahanDataset(
        comments=x,
        targets=y,
        tokenizer=tokenizer,
        max_len=max_len,
        label_list=label_list)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size) 
    
    
    
Loader = DataLoader()
train, valid, test= Loader.load()

tokenizer = BertTokenizer.from_pretrained(Settings.MODEL_NAME)


label_list = [0, 1]
train_data_loader = create_data_loader(train['comment'].to_numpy(), train['label'].to_numpy(), tokenizer, MAX_LEN, TRAIN_BATCH_SIZE, label_list)
valid_data_loader = create_data_loader(valid['comment'].to_numpy(), valid['label'].to_numpy(), tokenizer, MAX_LEN, VALID_BATCH_SIZE, label_list)
test_data_loader = create_data_loader(test['comment'].to_numpy(), None, tokenizer, MAX_LEN, TEST_BATCH_SIZE, label_list)




