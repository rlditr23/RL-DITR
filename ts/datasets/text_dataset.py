import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from pathlib import Path
from functools import lru_cache

from transformers import AutoTokenizer

@lru_cache(64)
def init_tokenizer(model_name_or_path):
    '''
    initialize tokenizer
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


class TextClfDataset(Dataset):
    def __init__(self, text, label, model_name_or_path, max_seq_len):
        super(TextClfDataset, self).__init__()
        
        if isinstance(model_name_or_path,str) or isinstance(model_name_or_path,Path):
            self._tokenizer = init_tokenizer(model_name_or_path)
        else:
            self._tokenizer = model_name_or_path
        self._max_seq_len = max_seq_len
        self.x = text
        self.y = label
        

    # @lru_cache(100000)
    def _convert(self, text):
        '''
        generate a corresponding id and complete/cut based on max_seq_len 
        '''
        tokens = self._tokenizer.tokenize(text)
        ids = self._tokenizer.convert_tokens_to_ids(tokens)
        if len(ids) < (self._max_seq_len-2):
            ids = ids + [0]*(self._max_seq_len-2-len(ids))
        else:
            ids = ids[:self._max_seq_len-2]
        ids = self._tokenizer.build_inputs_with_special_tokens(ids)
        ids = np.array(ids,dtype='int')
        return ids
        

    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        x = self._convert(self.x[idx])
        y = self.y[idx]
        return x, y


if __name__ == '__main__':
    df = pd.read_csv('data.grading.reg.csv')
    text = df['text']
    label = df['label']
    model_name_or_path = 'bert-base-chinese'
    max_seq_len = 256
    dataset = TextClfDataset(text, label, model_name_or_path, max_seq_len)
    for x,y in dataset:
        print(x.shape)
        print(y.shape)
        print(y)
        break

    dataloader = DataLoader(dataset,batch_size=2)
    for x,y in dataloader:
        print(x.shape)
        print(y.shape)
        print(y)
        break