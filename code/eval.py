import pandas as pd
import numpy as np
import json, os, re, time
import numpy as np
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertAdam
import torch
import torch.nn as nn
from model import MyModel, MyTextCNNModel, MyRCNNModel
from data_pre import MyDataset, get_dataloader
import os
import gc
import time, datetime
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(dim, 
            names,
            weight,
            batch_size, 
            pretrain_model_path,
            model_types=None):
    
    print('-' * 100)
    print('multi-models begin predicting ...')
    print('-' * 100)
    
    # read test data
    test_file = '../data/Dataset/test.csv'

    # data
    test_df = pd.read_csv(test_file)
    test_ids = test_df['id'].values.tolist()
    
    result_prob_tmp = torch.zeros((len(test_ids), 2))
    # load model
    for i, name in enumerate(names):
        
        # 3.17 add
        weight_ = weight[i]
        
        model_path = '../user_data/model_store/' + name + '.pkl'
        state = torch.load(model_path)

        # 3.10 add
        model_type = model_types[i]
        if model_type == 'mlp':
            test_iter = MyDataset(file=test_file, is_train=False, pretrain_model_path=pretrain_model_path[i])
            test_iter = get_dataloader(test_iter, batch_size, shuffle=False, drop_last=False)
            model = MyModel(dim=dim[i], pretrain_model_path=pretrain_model_path[i])
        
        elif model_type == 'cnn':
            test_iter = MyDataset(file=test_file, is_train=False, pretrain_model_path=pretrain_model_path[i])
            test_iter = get_dataloader(test_iter, batch_size, shuffle=False, drop_last=False)
            model = MyTextCNNModel(dim=dim[i], pretrain_model_path=pretrain_model_path[i])
        
        elif model_type == 'rcnn':
            test_iter = MyDataset(file=test_file, is_train=False, pretrain_model_path=pretrain_model_path[i])
            test_iter = get_dataloader(test_iter, batch_size, shuffle=False, drop_last=False)
            model = MyRCNNModel(dim=dim[i], pretrain_model_path=pretrain_model_path[i])
            
        model.to(device)
        model.load_state_dict(state['model_state'])
        model.eval()
        print('-'*20, 'model', i, '-'*20)
        print('load model:%s, loss:%.4f, e:%d, lr:%.7f, time:%d' %
                      (name, state['loss'], state['e'], state['lr'], state['time']))
        # predict
        with torch.no_grad():
            j = 0
            for batch in tqdm(test_iter):

                batch = [b.cuda() for b in batch]
                out = model(batch, task='eval')
                out = out.cpu() # gpu -> cpu
        
                if j == 0:
                    tmp = out # 初始化 tmp
                else:
                    tmp = torch.cat([tmp, out], dim=0) # 将之后的预测结果拼接到 tmp 中
                j += 1
        
        # 当前 模型预测完成
        print('model', i, 'predict finished!\n')
        # 3.17 按权重融合
        result_prob_tmp += (weight_ / len(names)) * tmp
        
        
        # 删除模型
        del model
        gc.collect()
        
        time.sleep(1)
    
    # 3.10 当前融合策略：prob 简单的取 avg
    _, result = torch.max(result_prob_tmp, dim=-1)
    result = result.numpy()
    
    # 3.16 update: label 0的prob 大于 3，就认为是 label=0
#     with open('tmp.txt', 'w', encoding='utf-8') as f:
#         for r in result_prob_tmp:
#             f.write(str(r) + '\n')
            
    # save result
    df = pd.DataFrame()
    df['id'] = test_ids
    df['label'] = result
    df.to_csv(("../prediction_result/result_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), encoding='utf-8', index=False)
#     df.to_csv('./result.csv', encoding='utf-8', index=False)


def startPredict():

    names = ['roberta_wwm_large_bs_16_ad',
             'roberta_wwm_large_bs_16_ad_lr_1_5', 
             'roberta_wwm_large_15251_epoch3',
             'roberta_wwm_large_15251_lr_1_5',
             'ure_24_bs_16_1e5_pgd',
             
             'roberta_large_textcnn_bs_16_1e5_pgd',
             'ure_24_textcnn_0310',
             
             'ure_24_rcnn_18128_0313']
    
    weight = [1.2, 1, 1, 1, 1.2,   1.2, 1.2,   1.2] # 3.27 0.9626
    dim = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
    roberta_large_path = '../data/External/pytorch-roberta-wwm-large/'
    pretrain_model_path = [roberta_large_path] * 8        
    model_types = ['mlp','mlp','mlp','mlp','mlp', 'cnn','cnn', 'rcnn']
    
    predict(dim=dim,
            names=names,
            weight=weight,
            batch_size=16,
            pretrain_model_path=pretrain_model_path,
            model_types=model_types)
    

if __name__ == '__main__':
    
    startPredict()