import torch
from torch import nn
import pandas as pd
import numpy as np


def cos_dis(X):    #计算余弦距离？？？
        """
        cosine distance
        :param X: (N, d)
        :return: (N, N)
        """
        X = nn.functional.normalize(X)   #归一化
        XT = X.transpose(0, 1)    #转置
        return torch.matmul(X, XT)  


def sample_ids(ids, k):       #提取k-1个样本和他本身
    """
    sample `k` indexes from ids, must sample the centroid node itself
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)
    sampled_ids = df.sample(k - 1, replace=True).values
    sampled_ids = sampled_ids.flatten().tolist()
    sampled_ids.append(ids[-1])  # must sample the centroid node itself
    return sampled_ids


def sample_ids_v2(ids, k):  #提取k个样本
    """
    purely sample `k` indexes from ids
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)
    # if(df.empty):                    
    #     empty_sample = []
    #     empty_sample = np.array(empty_sample)
    #     sampled_ids = empty_sample.flatten().tolist()
    # else:
    # np.savetxt("/data_sdd/datadrh/HOZ/test_data/sample_ids_v2_df.txt", df, fmt = '%s', delimiter = ',')
    sampled_ids = df.sample(k, replace=True).values   #随机提取k个样本，可重复采样
    sampled_ids = sampled_ids.flatten().tolist()
    return sampled_ids