#!/usr/bin/python
# -*- coding: utf-8 -*-
# Created with pycharm.
# User: sssd
# Date: 2018/3/20 10:54
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:    svm 实现中文情感分析

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


# 对每个句子的所有词向量取均值 (注意返回的是 numpy array)
def buildWorldVector(text, size, model):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


#  获取每句话的对应的向量
def getTrainVecs(train_x, test_x):
    # 训练 word2vec 模型
    n_dim = 300
    model = Word2Vec(size=n_dim, min_count=10)
    model.build_vocab(train_x)
    # Train the model over train_reviews (this may take several minutes)
    model.train(train_x)

    # 注意此时的 train_x 为 array, 但是每个元素存放的是 list
    # [buildWorldVector(z, n_dim, imdb_w2v) for z in train_x] 为list
    # 而其中 buildWorldVector(z, n_dim, imdb_w2v) 返回的 numpy array
    trainVecs = np.concatenate([buildWorldVector(z, n_dim, model) for z in train_x])
    np.save(os.path.join(config.BASH_PATH, "svm_data", "train_vecs.npy"), trainVecs)
    model.train(test_x)
    model.save(os.path.join(config.BASH_PATH, 'svm_data', 'wcmodel.pkl'))
    testVecs = np.concatenate([buildWorldVector(z, n_dim, model) for z in test_x])
    np.save(os.path.join(config.BASH_PATH, 'svm_data', 'test_vecs.npy'), testVecs)
    print('训练集输入的结构类型为：%s' % str(trainVecs.shape))
    print('测试集输入的结构类型为：%s' % str(testVecs.shape))

    # wc = lambda a: buildWorldVector([buildWorldVector(z, n_dim, imdb_w2v) for z in train_x], n_dim, imdb_w2v)
    # train_vec = np.concatenate(wc(train_x))
    # print (train_vec.shape)


# 从已经得到的文件中获取对应的数据
def getData():
    train_x = np.load(os.path.join(config.BASH_PATH, 'svm_data', 'train_vecs.npy'))
    train_y = np.load(os.path.join(config.BASH_PATH, 'svm_data', 'train_y.npy'))
    test_x = np.load(os.path.join(config.BASH_PATH, 'svm_data', 'test_vecs.npy'))
    test_y = np.load(os.path.join(config.BASH_PATH, 'svm_data', 'test_y.npy'))
    return train_x, train_y, test_x, test_y


# svm 训练模型
def svmTrain(train_x, train_y, test_x, test_y):
    svc = SVC(kernel='rbf', verbose=True)
    svc.fit(train_x, train_y)
    joblib.dump(svc, os.path.join(config.BASH_PATH, 'svm_data', 'svmModel.pkl'))
    print("训练模型时，测试集的预测精度为%s" % str(svc.score(test_x, test_y)))


# 根据词组获取对应的向量
def getPredictVecs(words):
    n_dim = 300
    word_vec_load = Word2Vec.load(os.path.join(config.BASH_PATH, 'svm_data', 'wcmodel.pkl'))
    world_vector = buildWorldVector(words, n_dim, word_vec_load)
    print("返回的词向量的数据类型为%s，结构为%s" % (str(type(world_vector)), str(world_vector.shape)))
    return world_vector


# 对单个语句进行情感判断
def svmPredice(string):
    words = jieba.lcut(string)
    print("结巴分词后的数据类型为%s，数据结构为%s" % (str(type(words)), str(len(words))))
    wordVecs = getPredictVecs(words)
    svmModel = joblib.load(os.path.join(config.BASH_PATH, 'svm_data', 'svmModel.pkl'))
    res = svmModel.predict(wordVecs)
    if int(res[0]) == 1:
        print("该文本是正面的！")
    else:
        print("该文本时负面的！")


if __name__ == '__main__':
    # train_x, test_x = loadfile()
    # getTrainVecs(train_x, test_x)
    # train_x, train_y, test_x, test_y = getData()
    # svmTrain(train_x, train_y, test_x, test_y)

    string = '第一次买plus的机器，感觉真的很不错啊，相比较SE的相机，iPhone 8 plus夜间拍照强了太多'
    svmPredice(string)
    pass
