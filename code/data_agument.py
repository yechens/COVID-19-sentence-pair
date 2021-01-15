import pandas as pd
import numpy as np
import random

def dataAug(train_file='../data/Dataset/train.csv',
            dev_file='../data/Dataset/dev.csv'):
    '''获取数据增强的函数
       增强方式：数据传递，例如A与B相似，A与C相似 => B与C相似
    '''
    print('data augment ... ')
    # train-aug
    df_train = pd.read_csv(train_file)
    query1 = df_train['query1'].values.tolist()
    query2 = df_train['query2'].values.tolist()
    label = df_train['label'].values.tolist()
    # 采集额外的正负样本
    extra = []
    extra_neg = []
    tmp = []
    tmp_neg = []
    pre = query1[0]
    for q1, q2, l in zip(query1, query2, label):
        if q1==pre:
            if l==1:
                tmp.append(q2)
            else:
                tmp_neg.append(q2)
        else:
            if len(tmp)>=2:
                tmp = list(set(tmp))
                extra.append(tmp)
                if tmp and tmp_neg:
                    t1 = random.sample(tmp, 1)[0]
                    t2 = random.sample(tmp_neg, 1)[0]
                    extra_neg.append([t1, t2])
            pre = q1
            tmp = []
            tmp_neg = []
            if l==1: # 只有是相似问时，才加入 tmp
                tmp.append(q2)
            else:
                tmp_neg.append(q2)
    
    extra_query1, extra_query2, extra_label = [], [], []
    neg_extra_query1, neg_extra_query2, neg_extra_label = [], [], []

    for item in extra:
        if len(item) == 2:
            extra_query1.append(item[0])
            extra_query2.append(item[1])
            extra_label.append(1)
        else: # 有2条以上的相似问句，两两组合匹配
            tot = len(item)
            for i in range(tot):
                for j in range(i+1, tot):
                    x1,x2 = item[i],item[j]
                    extra_query1.append(item[0])
                    extra_query2.append(item[1])
                    extra_label.append(1)

    for item in extra_neg:
        neg_extra_query1.append(item[0])
        neg_extra_query2.append(item[1])
        neg_extra_label.append(0)

#     print('tot postive extra query:', len(extra_query1))
#     print('tot passive extra query:', len(neg_extra_query1))
    
    # dev-aug
    df_dev = pd.read_csv(dev_file)
    query11 = df_dev['query1'].values.tolist()
    query22 = df_dev['query2'].values.tolist()
    labell = df_dev['label'].values.tolist()
    # sample
    extra = []
    extra_neg = []
    tmp = []
    tmp_neg = []
    pre = query1[0]
    for q1, q2, l in zip(query11, query22, labell):
        if q1==pre:
            if l==1:
                tmp.append(q2)
            else:
                tmp_neg.append(q2)
        else:
            if len(tmp)>=2:
                tmp = list(set(tmp))
                extra.append(tmp)
                # 3.5 从当前正负样本中，分别采样一条，构成一对新的负样本
                if tmp and tmp_neg:
                    t1 = random.sample(tmp, 1)[0]
                    t2 = random.sample(tmp_neg, 1)[0]
                    extra_neg.append([t1, t2])
            pre = q1
            tmp = []
            tmp_neg = []
            if l==1: # 只有是相似问时，才加入 tmp
                tmp.append(q2)
            else:
                tmp_neg.append(q2)
#     print('dev pos extra:', len(extra))
#     print('dev neg extra:', len(extra_neg))
    
    # extra_query1, extra_query2, extra_label = [], [], []
#     print('extra pos query before dev:', len(extra_query1))
#     print('extra neg query before dev:', len(neg_extra_query1))

    for item in extra:
        if len(item) == 2:
            extra_query1.append(item[0])
            extra_query2.append(item[1])
            extra_label.append(1)
        else: # 有2条以上的相似问句，两两组合匹配
            tot = len(item)
            for i in range(tot):
                for j in range(i+1, tot):
                    x1,x2 = item[i],item[j]
                    extra_query1.append(item[0])
                    extra_query2.append(item[1])
                    extra_label.append(1)

    for item in extra_neg:
        neg_extra_query1.append(item[0])
        neg_extra_query2.append(item[1])
        neg_extra_label.append(0)

#     print('extra pos query after dev:', len(extra_query1), len(extra_query2), len(extra_label))
#     print('extra neg query after dev:', len(neg_extra_query1), len(neg_extra_query2), len(neg_extra_label))
    
    # save new dataset
    tot1, tot2, totl = [], [], []
    # train
    for q1,q2,l in zip(query1, query2, label):
        tot1.append(q1); tot2.append(q2); totl.append(l)
#     print('tot:', len(tot1))
    # train + dev
    for q1,q2,l in zip(query11, query22, labell):
        tot1.append(q1); tot2.append(q2); totl.append(l)
#     print('tot:', len(tot1))
    # train + dev + pos_extra_query
    for q1,q2,l in zip(extra_query1, extra_query2, extra_label):
        tot1.append(q1); tot2.append(q2); totl.append(l)
#     print('tot:', len(tot1))
    # 统计正负样本数量；最后再加入 extra_neg_query
#     print(pd.Series(totl).value_counts())
    
    for t1, t2 in zip(neg_extra_query1, neg_extra_query2):
        tot1.append(t1)
        tot2.append(t2)
        totl.append(0)
#     print('add negtive data!')
#     print(pd.Series(totl).value_counts())
    
    # 在增强的数据集上重新划分 train / dev
    np.random.seed(42)
    t1,t2,tl = [],[],[]
    d1,d2,dl = [],[],[]
    random_idx = np.arange(len(tot1))
    np.random.shuffle(random_idx)
    np.random.shuffle(random_idx)
    # 保留 800 做线下 dev
    for idx in random_idx[:-800]:
        t1.append(tot1[idx]); t2.append(tot2[idx]); tl.append(totl[idx])

    for idx in random_idx[-800:]:
        d1.append(tot1[idx]); d2.append(tot2[idx]); dl.append(totl[idx])
    print('new train:', len(t1), len(t2), len(tl))
    print('new dev:', len(d1), len(d2), len(dl))
    # save to file
    my_train = pd.DataFrame()
    my_train['id'] = range(0, len(t1))
    my_train['query1'] = t1
    my_train['query2'] = t2
    my_train['label'] = tl
    my_train.to_csv('../data/Dataset/my_train.csv', encoding='utf-8', index=False)

    my_dev = pd.DataFrame()
    my_dev['id'] = range(0, len(d1))
    my_dev['query1'] = d1
    my_dev['query2'] = d2
    my_dev['label'] = dl
    my_dev.to_csv('../data/Dataset/my_dev.csv', encoding='utf-8', index=False)
    
    print('data augment over!\n')
    
    
if __name__ == '__main__':
    
    dataAug()