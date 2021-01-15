# 此脚本仅存放数据读取类
import pandas as pd
import numpy as np
import json, os, re, time
import numpy as np
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertAdam
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from helper import LabelSmoothingLoss, pad, PGD


class MyDataset(Dataset):
    def __init__(self, 
                 file, is_train=True, 
                 sample=False, 
                 pretrain_model_path='',
                 add_edit_dist=False):
        self.is_train = is_train
        tokenizer = BertTokenizer.from_pretrained(pretrain_model_path, do_lower_case=False, cache_dir=None)
        
        df = pd.read_csv(file)
        querys1 = df['query1'].values.tolist()
        querys2 = df['query2'].values.tolist()
        
        self.inputs, self.input_types = [], []
        print('deal with query ...')
        for q1, q2 in zip(querys1, querys2):
            
            input_tmp, input_type = [], []
            q1 = tokenizer.tokenize(q1)
            q1 = tokenizer.convert_tokens_to_ids(['[CLS]'] + q1 + ['[SEP]'])
            input_tmp += q1
            input_type += [0 for _ in range(len(q1))]
            
            q2 = tokenizer.tokenize(q2)
            q2 = tokenizer.convert_tokens_to_ids(q2 + ['[SEP]'])
            input_tmp += q2
            input_type += [1 for _ in range(len(q2))]

            self.inputs.append(input_tmp)
            self.input_types.append(input_type)
        
        if is_train:
            self.labels = df['label'].values.tolist()
            
        # 计算最大长度，方便调用 __getitem__ 函数时进行填充
        self.max_input_len = max([len(s) for s in self.inputs])
        print('max_input_len:', self.max_input_len)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, item):
        if self.is_train: # 如果是训练
            # pad 填充操作
            input_item = pad(self.inputs[item], self.max_input_len, 0)
            input_type_item = pad(self.input_types[item], self.max_input_len, 0)
            label_item = self.labels[item]
            return torch.LongTensor(input_item),\
                   torch.LongTensor(input_type_item), \
                   torch.LongTensor([label_item])
        else:
            input_item = pad(self.inputs[item], self.max_input_len, 0)
            input_type_item = pad(self.input_types[item], self.max_input_len, 0)
            return torch.LongTensor(input_item),\
                   torch.LongTensor(input_type_item)

def get_dataloader(dataset, batch_size, shuffle, drop_last):
    data_iter = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return data_iter
