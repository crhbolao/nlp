#!/usr/bin/python
# Created with pycharm.
# File Name: Lstm_pre
# User: sssd
# Date: 2018/3/21 15:47
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:  LSTM 实现中文情感分析

from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml
import jieba
import pandas as pd
import sentiment_analysis.config as config
import yaml
import os

# word2vec 的参数配置
vocab_dim = 100
n_exposures = 10
window_size = 7
cpu_count = multiprocessing.cpu_count()
n_iterations = 1  # ideally more..

# 设置向量参数
maxlen = 100


# 加载数据文件，构建数据输入和输出
def loadFile():
    neg = pd.read_excel(os.path.join(config.BASH_PATH, 'data', 'neg.xls'), header=None, index=None)
    pos = pd.read_excel(os.path.join(config.BASH_PATH, 'data', 'pos.xls'), header=None, index=None)
    data = np.concatenate((pos[0], neg[0]))
    label = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg[0]), dtype=int)))
    print("输入的数据类型为：%s，格式为：%s。输出的数据类型为：%s，格式为：%s。" % (str(type(data)), str(data.shape), str(type(label)), str(label.shape)))
    return data, label


# 对文本进行分析，并去掉其中的换行符
def tokenizer(text):
    data = [jieba.lcut(line.replace("\n", "")) for line in text]
    print("结巴分此后的数据类型为：%s，数据长度为%s。" % (str(type(data)), str(len(data))))
    return data


# 构建词典，返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def buildDic(model=None, words=None):
    if (model is not None) and (words is not None):
        # 初始化一个词典
        dict = Dictionary()
        # model.vocab.keys() 为 word2vec 中所有的词，设置 allow_update=True 则每个词出现一个，频率就会增加一次
        dict.doc2bow(model.vocab.keys(), allow_update=True)
        # 重新生成字典：key 是单词，value 是单词对应的下标。其中 k 为下标索引，v 为 字典中包含的词，
        w2indx = {v: k + 1 for k, v in dict.items()}
        # key 是单词，value 是对应的词向量
        w2vec = {word: model[word] for word in w2indx.keys()}

        # 解析数据获取数据单词对应的索引
        def parseDataset(words):
            data = []
            for sentence in words:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parseDataset(words)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec, combined
    else:
        print("模型或数据导入失败")


# 训练 word2vec 模型，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def wcTrain(words):
    model = Word2Vec(size=vocab_dim, min_count=n_exposures, window=window_size, workers=cpu_count, iter=n_iterations)
    model.build_vocab(words)
    model.train(words)
    model.save(os.path.join(config.BASH_PATH, 'lstm_data', 'Word2vec_model.pkl'))
    index_dict, word_vectors, combined = buildDic(model=model, words=words)
    return index_dict, word_vectors, combined

# 训练lstm
def train():
    print("开始导入数据...")
    data, label = loadFile()
    print("开始进行分词...")
    jieba_data = tokenizer(data)
    print("开始训练word2vec...")
    index_dict, word_vectors, combined = wcTrain(jieba_data)

    '''
    开始在这写
    '''


if __name__ == '__main__':
    train()
    pass
