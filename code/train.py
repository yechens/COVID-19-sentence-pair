# 3.12: 此脚本主要用于线下训练/评测模型
import pandas as pd
import numpy as np
import json, os, re, time, gc
import numpy as np
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertAdam
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
# 导入所有模型
from model import MyModel, MyTextCNNModel, MyRCNNModel
# 导入所有数据读取类
from data_pre import MyDataset, get_dataloader
from data_agument import dataAug
# 导入所有工具类
from helper import LabelSmoothingLoss, pad, PGD, FGM, count_right_num

# 确保代码“兼容” cpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print('*** n_gpu:', n_gpu, '***')


# 训练、评测
def train(batch_size=16, 
          pretrain_model_path='', 
          name='', 
          model_type='mlp',
          dim=1024, 
          lr=1e-5, 
          epoch=12, 
          smoothing=0.05,
          sample=False, 
          open_ad='',
          dialog_name='xxx'):

    if not pretrain_model_path or not name:
        assert 1==-1
        
#     print('\n********** model type:', model_type, '**********')
#     print('batch_size:', batch_size)
    
    # load dataset
    train_file = '../data/Dataset/my_train.csv'
    dev_file = '../data/Dataset/my_dev.csv'
    
    train_num = len(pd.read_csv(train_file).values.tolist())
    val_num = len(pd.read_csv(dev_file).values.tolist())
    print('train_num: %d, dev_num: %d' % (train_num, val_num))

    # 选择模型
    if model_type in ['siam', 'esim', 'sbert']:
        assert 1==-1

    else:
        train_iter = MyDataset(file=train_file, is_train=True, sample=sample, pretrain_model_path=pretrain_model_path)
        train_iter = get_dataloader(train_iter, batch_size, shuffle=True, drop_last=True)
        dev_iter = MyDataset(file=dev_file, is_train=True, sample=sample, pretrain_model_path=pretrain_model_path)
        dev_iter = get_dataloader(dev_iter, batch_size, shuffle=False, drop_last=False)
    
        if model_type == 'mlp':
            model = MyModel(dim=dim, pretrain_model_path=pretrain_model_path, smoothing=smoothing)
            
        elif model_type == 'cnn':
            model = MyTextCNNModel(dim=dim, pretrain_model_path=pretrain_model_path, smoothing=smoothing)
            
        elif model_type == 'rcnn':
            model = MyRCNNModel(dim=dim, pretrain_model_path=pretrain_model_path, smoothing=smoothing)
            
    model.to(device)
    model_param_num = 0
    
    ##### 3.24 muppti-gpu-training
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    for p in model.parameters():
        if p.requires_grad:
            model_param_num += p.nelement()
    print('param_num:%d\n' % model_param_num)
    
    # 加入对抗训练，提升泛化能力；但是训练速度明显变慢 (插件式调用)
    # 3.12 change to FGM 更快！
    if open_ad == 'fgm':
        fgm = FGM(model)
    elif open_ad == 'pgd':
        pgd = PGD(model)
        K = 3

    # model-store-path
    model_path = '../user_data/model_store/' + name + '.pkl' # 输出模型默认存放在当前路径下
    state = {}
    time0 = time.time()
    best_loss = 999
    early_stop = 0 
    for e in range(epoch):
        print("*" * 100)
        print("Epoch:", e)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=lr,
                             warmup=0.05,
                             t_total=len(train_iter)) # 设置优化器
        train_loss = 0
        train_c = 0
        train_right_num = 0
    
        model.train() # 将模型设置成训练模式（Sets the module in training mode）
        print('training..., %s, e:%d, lr:%7f' % (name, e, lr))
        for batch in tqdm(train_iter): # 每一次返回 batch_size 条数据
            
            optimizer.zero_grad() # 清空梯度
            batch = [b.cuda() for b in batch] # cpu -> GPU
            
            # 正常训练
            labels = batch[-1].view(-1).cpu().numpy()
            loss, bert_enc = model(batch, task='train', epoch=epoch) # 进行前向传播，真正开始训练；计算 loss
            right_num = count_right_num(bert_enc, labels)
            
            # multi-gpu training!
            if n_gpu > 1:
                loss = loss.mean()
            
            loss.backward() # 反向传播计算参数的梯度
            
            if open_ad == 'fgm':
                # 对抗训练
                fgm.attack() # 在embedding上添加对抗扰动
                
                if model_type == 'multi-task': loss_adv, _, _ = model(batch, task='train')
                else: loss_adv, _ = model(batch, task='train')

                if n_gpu > 1:
                    loss_adv = loss_adv.mean()
                
                loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore() # 恢复embedding参数
        
            elif open_ad == 'pgd':
                pgd.backup_grad()
                # 对抗训练
                for t in range(K):
                    pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K-1:
                        optimizer.zero_grad()
                    else:
                        pgd.restore_grad()
                        
                    if model_type == 'multi-task': loss_adv, _, _ = model(batch, task='train')
                    else: loss_adv, _ = model(batch, task='train')

                    if n_gpu > 1:
                        loss_adv = loss_adv.mean()

                    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore() # 恢复embedding参数
            
            optimizer.step() # 更新参数

            train_loss += loss.item() # loss 求和
            train_c += 1
            train_right_num += right_num

        val_loss = 0
        val_c = 0
        val_right_num = 0

        model.eval() 
        print('eval...')
        with torch.no_grad(): # 不进行梯度的反向传播
            for batch in tqdm(dev_iter): # 每一次返回 batch_size 条数据
                batch = [b.cuda() for b in batch]
                
                labels = batch[-1].view(-1).cpu().numpy()
                loss, bert_enc = model(batch, task='train', epoch=epoch) # 进行前向传播，真正开始训练；计算 loss
                right_num = count_right_num(bert_enc, labels)
                
                if n_gpu > 1:
                    loss = loss.mean()
        
                val_c += 1
                val_loss += loss.item()
                val_right_num += right_num
                                
        train_acc = train_right_num / train_num
        val_acc = val_right_num / val_num
        
        print('train_acc: %.4f, val_acc: %.4f' % (train_acc, val_acc))
        print('train_loss: %.4f, val_loss: %.4f, time: %d' % (train_loss/train_c, val_loss/val_c, time.time()-time0))
        
        if val_loss / val_c < best_loss:
            early_stop = 0
            best_loss = val_loss / val_c
            best_acc = val_acc
            
            # 3.24 update 多卡训练时模型保存避坑:
            model_to_save = model.module if hasattr(model, 'module') else model
            state['model_state'] = model_to_save.state_dict()
            state['loss'] = val_loss / val_c
            state['acc'] = val_acc
            state['e'] = e
            state['time'] = time.time() - time0
            state['lr'] = lr
                
            torch.save(state, model_path)
            
            best_epoch = e
            cost_time = time.time() - time0
            tmp_train_acc = train_acc
            best_model = model
            
        else:
            early_stop += 1
            if early_stop == 2:
                break
            
            model = best_model
            lr = lr * 0.5
        print("best_loss:", best_loss)
    
    # 3.12 add 打印显示最终的最优结果
    print('-' * 30)
    print('best_epoch:', best_epoch, 'best_loss:', best_loss, 'best_acc:', best_acc, 'reach time:', cost_time, '\n')
    
    # model-clean
    del model
    gc.collect()
    
    # 实验结果写入日志
    with open('../user_data/model_dialog/' + dialog_name + '.out', 'w', encoding='utf-8') as f:
        f.write('*** model name:' + dialog_name + ' ***\n')
        f.write('best dev acc:' + str(best_acc) + '\n')
        f.write('best loss:' + str(best_loss) + '\n')
        f.write('best_epoch:' + str(best_epoch) + '\n')
        f.write('train acc:' + str(tmp_train_acc) + '\n')
        f.write('lr:' + str(state['lr']) + '\n')
        f.write('time:' + str(cost_time) + '\n')
        
        
def startTrain():
    
    # 一共要训练8个model
    # 3.28 实际训练时，ure统一替换成roberta
    # kd_roberta_wwm_large_bs_16_ad -> roberta_wwm_large_bs_16_ad_lr_1_5
    names = ['roberta_wwm_large_bs_16_ad',
             'roberta_wwm_large_bs_16_ad_lr_1_5', 
             'roberta_wwm_large_15251_epoch3',
             'roberta_wwm_large_15251_lr_1_5',
             'ure_24_bs_16_1e5_pgd',
             
             'roberta_large_textcnn_bs_16_1e5_pgd',
             'ure_24_textcnn_0310',
             
             'ure_24_rcnn_18128_0313']
    
    roberta_large_path = '../data/External/pytorch-roberta-wwm-large/'
    ure_24_path = '../data/External/pytorch-24-URE/'
    # 定义不同 model 的参数
    lr_lists = [1e-5, 1.5e-5, 1e-5, 1.5e-5, 1e-5,  1e-5, 1e-5,  1e-5]
    batch_sizes = [16, 16, 24, 24, 16,  16, 24,  16]
    ad_policies = ['pgd', 'pgd', 'fgm', 'fgm', 'pgd',  'pgd', 'fgm',  'pgd']
    model_types = ['mlp','mlp','mlp','mlp','mlp', 'cnn','cnn', 'rcnn']
    smoothings = [0.1, 0.05, 0.1, 0.1, 0.05,  0.05, 0.05,  0.05]
    pretrain_model_paths = [
        roberta_large_path,
        roberta_large_path,
        roberta_large_path,
        roberta_large_path,
        ure_24_path,
        roberta_large_path,
        ure_24_path,
        ure_24_path
    ]
    
    time1 = time.time()
    idx = 0
    for name, lr, open_ad, batch_size, model_type, smoothing, pretrain_model_path in zip(names, lr_lists, ad_policies, batch_sizes, model_types, smoothings, pretrain_model_paths):
        
        print('#' * 100)
        print('begin training model', idx)
        print('name:', name, '| lr:', lr, '| open_ad:', open_ad, '| batch_size:', batch_size, '| model_type:', model_type, '| smoothing:', smoothing)
        
        train(batch_size=batch_size, 
              pretrain_model_path=pretrain_model_path,
              name=name, 
              model_type=model_type,
              dim=1024,
              lr=lr,
              epoch=12,
              smoothing=smoothing,
              open_ad=open_ad,
              dialog_name=name)
        
        idx += 1
        
        time.sleep(1)
        
    time2 = time.time()
    print('+' * 100)
    print('total training time:', time2 - time1)
    print('+' * 100)
       
        
if __name__ == '__main__':
    
    startTrain()
    
