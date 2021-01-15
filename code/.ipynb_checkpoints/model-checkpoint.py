# 此脚本仅存放模型的定义
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
from helper import LabelSmoothingLoss, pad, PGD, FGM

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyModel(nn.Module):
    def __init__(self, 
                 dim=768, 
                 pretrain_model_path=None,
                 add_edit_dist=False,
                 smoothing=0.05):
        super(MyModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(pretrain_model_path, cache_dir=None)
        
        self.linear1 = nn.Linear(dim, 2)    
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters() # 层初始化
        self.label_smooth_loss = LabelSmoothingLoss(classes=2, smoothing=smoothing)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0.0)

    def forward(self, batch, task='eval', epoch=0):
        '''train: 完成了模型预测输出 + loss计算求和 两个过程
           valid: 完成了模型预测输出
        '''
        if task == 'train':
            
            inputs, input_types, labels = batch
            bert_mask = torch.ne(inputs, 0) # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）
            # bert_enc：词向量   pooled_out：句向量
            bert_enc, _ = self.bert(inputs, token_type_ids=input_types, attention_mask=bert_mask, output_all_encoded_layers=False) 
            # 3.9 先 dropout
            bert_enc = self.dropout(bert_enc)
                
            ##### 3.10 大 bug 修复：取 mean 之前，应该先把 padding 部分的特征去除！！！
            mask_2 = bert_mask # 其余等于 1 的部分，即有效的部分                
            mask_2_expand = mask_2.unsqueeze_(-1).expand(bert_enc.size()).float()
            sum_mask = mask_2_expand.sum(dim=1) # 有效的部分“长度”求和
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            bert_enc = torch.sum(bert_enc * mask_2_expand, dim=1) / sum_mask
            #####
                
            bert_enc = self.linear1(bert_enc)
            
            # 3.5 add label smoothing
            loss = self.label_smooth_loss(bert_enc, labels.view(-1))
            return loss, bert_enc

        else:
            inputs, input_types = batch
            # bert enc
            bert_mask = torch.ne(inputs, 0) # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）
            # bert_enc：词向量   pooled_out：句向量
            bert_enc, _ = self.bert(inputs, token_type_ids=input_types, attention_mask=bert_mask, output_all_encoded_layers=False) 

            #####
            mask_2 = bert_mask # 其余等于 1 的部分，即有效的部分                
            mask_2_expand = mask_2.unsqueeze_(-1).expand(bert_enc.size()).float()
            sum_mask = mask_2_expand.sum(dim=1) # 有效的部分“长度”求和
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            bert_enc = torch.sum(bert_enc * mask_2_expand, dim=1) / sum_mask
            #####
                
            bert_enc = self.linear1(bert_enc)
            out = torch.softmax(bert_enc, dim=-1) # 不要忘了加激活函数！
            return out

            
            
# 3.10 add BERT+TextCNN model
class MyTextCNNModel(nn.Module):
        
    def __init__(self, 
                 dim=768, 
                 pretrain_model_path=None,
                 add_edit_dist=False,
                 pool_way='avg',
                 weight=[1.,1.],
                 filter_num=128,
                 filter_sizes=[2,3,4],
                 smoothing=0.05):
        
        super(MyTextCNNModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, cache_dir=None)
        self.dropout = nn.Dropout(0.1)
        
        # textcnn
        class_num = 2
        chanel_num = 1

        # 3.13 test
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, dim)) for size in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)
        
        self.reset_parameters() # fc层初始化
        self.label_smooth_loss = LabelSmoothingLoss(classes=2, smoothing=smoothing)
        
        self.pool_way = pool_way
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, batch, task='eval', epoch=0):
        '''train: 完成了模型预测输出 + loss计算求和 两个过程
           valid: 完成了模型预测输出
        '''
        if task == 'train':

            inputs, input_types, labels = batch
            bert_mask = torch.ne(inputs, 0) # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）
            # bert_enc：词向量   pooled_out：句向量
            bert_enc, _ = self.bert(inputs, token_type_ids=input_types, attention_mask=bert_mask, output_all_encoded_layers=False) 
            bert_enc = self.dropout(bert_enc)
            
            # textcnn
            x = bert_enc.unsqueeze(1) # conv2d 需要接收 4维 的输入
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
            
            if self.pool_way == 'max':
                x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
            elif self.pool_way == 'avg':
                x = [F.avg_pool1d(item, item.size(2)).squeeze(2) for item in x]
        
            x = torch.cat(x, 1)
            x = self.dropout(x)
            logits = self.fc(x)
            
            # 3.5 add label smoothing
            loss = self.label_smooth_loss(logits, labels.view(-1))
            return loss, logits

        else:
            # new
            inputs, input_types = batch
            # bert enc
            bert_mask = torch.ne(inputs, 0) # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）
            # bert_enc：词向量   pooled_out：句向量
            bert_enc, _ = self.bert(inputs, token_type_ids=input_types, attention_mask=bert_mask, output_all_encoded_layers=False) 
            
            # textcnn
            x = bert_enc.unsqueeze(1) # conv2d 需要接收 4维 的输入
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
            if self.pool_way == 'max':
                x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
            elif self.pool_way == 'avg':
                x = [F.avg_pool1d(item, item.size(2)).squeeze(2) for item in x]
            x = torch.cat(x, 1)
            logits = self.fc(x)
            
            out = torch.softmax(logits, dim=-1) # 不要忘了加激活函数！
            return out
        

class MyRCNNModel(nn.Module):
    '''双向 RNN + 池化层(CNN)
    '''
    def __init__(self, 
				 dim=768, 				 
                 pretrain_model_path=None,
                 add_edit_dist=False,
                 weight=[1,1],
                 hidden_size=128,
                 smoothing=0.05):

        super(MyRCNNModel, self).__init__()
        
        print('hidden_size:', hidden_size)
        
        self.bert = BertModel.from_pretrained(pretrain_model_path, cache_dir=None)
        self.dropout = nn.Dropout(0.1)
        self.label_smooth_loss = LabelSmoothingLoss(classes=2, smoothing=smoothing)
        
        self.hidden_size = hidden_size
        self.embedding_length = dim
        dropout_in_lstm = 0.5
        
        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size, dropout=dropout_in_lstm, bidirectional=True)
        self.W2 = nn.Linear(2 * self.hidden_size + self.embedding_length, self.hidden_size)
        self.label = nn.Linear(self.hidden_size, 2)
		
    def forward(self, batch, task='eval', epoch=0):

        if task == 'train':
            inputs, input_types, labels = batch

            bert_mask = torch.ne(inputs, 0) # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）
            # bert_enc：词向量   pooled_out：句向量
            bert_enc, _ = self.bert(inputs, token_type_ids=input_types, attention_mask=bert_mask, output_all_encoded_layers=False) 
            bert_enc = self.dropout(bert_enc)

            bert_enc = bert_enc.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)

            h_0 = torch.zeros(2, bert_enc.size(1), self.hidden_size).to(device)
            c_0 = torch.zeros(2, bert_enc.size(1), self.hidden_size).to(device)

            output, (final_hidden_state, final_cell_state) = self.lstm(bert_enc, (h_0, c_0))

            final_encoding = torch.cat((output, bert_enc), 2).permute(1, 0, 2)
            y = self.W2(final_encoding) # y.size() = (batch_size, num_sequences, hidden_size)
            y = y.permute(0, 2, 1) # y.size() = (batch_size, hidden_size, num_sequences)
            y = F.max_pool1d(y, y.size()[2]) # y.size() = (batch_size, hidden_size, 1)
            y = y.squeeze(2)
            logits = self.label(y)
  
            loss = self.label_smooth_loss(logits, labels.view(-1))
        
            return loss, logits

        else:
            inputs, input_types = batch
            
            bert_mask = torch.ne(inputs, 0) # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）
            # bert_enc：词向量   pooled_out：句向量
            bert_enc, _ = self.bert(inputs, token_type_ids=input_types, attention_mask=bert_mask, output_all_encoded_layers=False) 

            bert_enc = bert_enc.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)

            h_0 = torch.zeros(2, bert_enc.size(1), self.hidden_size).to(device)
            c_0 = torch.zeros(2, bert_enc.size(1), self.hidden_size).to(device)

            output, (final_hidden_state, final_cell_state) = self.lstm(bert_enc, (h_0, c_0))

            final_encoding = torch.cat((output, bert_enc), 2).permute(1, 0, 2)
            y = self.W2(final_encoding) # y.size() = (batch_size, num_sequences, hidden_size)
            y = y.permute(0, 2, 1) # y.size() = (batch_size, hidden_size, num_sequences)
            y = F.max_pool1d(y, y.size()[2]) # y.size() = (batch_size, hidden_size, 1)
            y = y.squeeze(2)
        
            logits = self.label(y)
            logits = torch.softmax(logits, dim=-1)
            return logits
