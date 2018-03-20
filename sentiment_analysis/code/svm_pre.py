#!/usr/bin/python
# -*- coding: utf-8 -*-
# Created with pycharm.
# User: sssd
# Date: 2018/3/20 10:54
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:

from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
import sentiment_analysis.config as config
import os

from sklearn.externals import joblib
from sklearn.svm import SVC


# 加载文件,导入数据,分词
def loadfile():
    neg = pd.read_excel(os.path.join(config.BASH_PATH, "data", "neg.xls"), header=None, index_col=None)
    pos = pd.read_excel(os.path.join(config.BASH_PATH, "data", "pos.xls"), header=None, index_col=None)

    # 定义lamada 表达式
    cw = lambda x: list(jieba.cut(x))

    # 将结巴分词后的结果存在在对应 name 中。
    neg['words'] = neg[0].apply(cw)
    pos['words'] = pos[0].apply(cw)

    # 构建样本标签y
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
    trian_x, test_x, train_y, test_y = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)
    # 将随机划分的标签数据保存
    np.save(os.path.join(config.BASH_PATH, 'svm_data', 'train_y.npy', ), train_y)
    np.save(os.path.join(config.BASH_PATH, 'svm_data', 'test_y.npy', ), test_y)
    return trian_x, test_x


# 对每个句子的所有词向量取均值
def buildWorldVector(text, size, model):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 计算词向量
def getTrainVecs(train_x, test_x):
    n_dim = 200
    model = Word2Vec(size=n_dim, min_count=1)
    model.build_vocab(train_x)
    model.train(train_x, total_examples=model.corpus_count, epochs=model.iter)
    wc = lambda a: buildWorldVector(a, n_dim, model)
    train_vec = wc(train_x)
    print(train_vec.shape)
    test_vec = wc(test_x)
    print(test_vec.shape)


if __name__ == '__main__':
    train_x, test_x = loadfile()
    getTrainVecs(train_x, test_x)
    pass
