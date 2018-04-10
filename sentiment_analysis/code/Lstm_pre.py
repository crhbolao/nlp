#!/usr/bin/python
# Created with pycharm.
# File Name: Lstm_pre
# User: sssd
# Date: 2018/3/21 15:47
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:  LSTM 实现中文情感分析 ，并让 python 程序和java程序之间通过RPC通信交互！


import os

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
from xmlrpc.server import SimpleXMLRPCServer


class LSTM:

    def init(self):
        print('加载训练好的数据模型...')
        # 加载 lstm 的网络模型
        with open(os.path.join(config.BASH_PATH, 'lstm_data', 'lstm.yml'), 'r') as f:
            modelStr = yaml.load(f)
        model = model_from_yaml(modelStr)

        print("加载训练好的模型参数...")
        # 加载 lstm 模型的权值参数
        model.load_weights(os.path.join(config.BASH_PATH, 'lstm_data', 'lstm.h5'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def __init__(self):
        # word2vec 的参数配置
        self.vocab_dim = 100
        self.n_exposures = 10
        self.window_size = 7
        self.cpu_count = multiprocessing.cpu_count()
        self.n_iterations = 1  # ideally more..

        # LSTM 相关参数
        self.maxlen = 100  # 输入数据的维度
        self.vocab_dim = 100  # 权值向量的维度
        self.input_length = 100
        self.batch_size = 50  # 批数据大小
        self.n_epoch = 10  # 设置的数据批次

        print('加载训练好的数据模型...')
        # 加载 lstm 的网络模型
        with open(os.path.join(config.BASH_PATH, 'lstm_data', 'lstm.yml'), 'r') as f:
            modelStr = yaml.load(f)
        self.model = model_from_yaml(modelStr)

        print("加载训练好的模型参数...")
        # 加载 lstm 模型的权值参数
        self.model.load_weights(os.path.join(config.BASH_PATH, 'lstm_data', 'lstm.h5'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def loadFile(self):
        '''
        加载训练数据
        :return:    合并后的数据，以及数据对应的标签
        '''
        neg = pd.read_excel(os.path.join(config.BASH_PATH, 'data', 'neg.xls'), header=None, index=None)
        pos = pd.read_excel(os.path.join(config.BASH_PATH, 'data', 'pos.xls'), header=None, index=None)
        data = np.concatenate((pos[0], neg[0]))
        label = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg[0]), dtype=int)))
        print("输入的数据类型为：%s，格式为：%s。输出的数据类型为：%s，格式为：%s。" % (str(type(data)), str(data.shape), str(type(label)), str(label.shape)))
        return data, label

    def tokenizer(self, text):
        '''
        利用结巴分词对文本进行分词
        :param text:   所有文本
        :return:       分词后的所有文本
        '''
        # “[]” 返回的是list。其中将换行符替换掉。
        data = [jieba.lcut(line.replace("\n", "")) for line in text]
        print("结巴分此后的数据类型为：%s，数据长度为%s。" % (str(type(data)), str(len(data))))
        return data

    def buildDic(self, model=None, words=None):
        '''
        构建词典，
        :param model:   word2vec模型
        :param words:   结巴分词后所有的文本内容
        :return:        返回每个词语的索引（词语-索引），词向量（词语-向量），以及每个句子所对应的词语索引（下标索引）
        '''
        if (model is not None) and (words is not None):
            # 初始化一个词典
            dict = Dictionary()
            # model.vocab.keys() 为 word2vec 中所有的词，设置 allow_update=True 则每个词出现一个，频率就会增加一次
            # 转换为词袋模型
            dict.doc2bow(model.vocab.keys(), allow_update=True)
            # 重新生成字典：key 是单词，value 是单词对应的下标。其中 k 为下标索引，v 为 字典中包含的词，
            w2indx = {v: k + 1 for k, v in dict.items()}
            # key 是单词，value 是对应的词向量
            w2vec = {word: model[word] for word in w2indx.keys()}

            # 获取一句话所对应的词语索引
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
            # 对长短不同的时序统一维度。
            combined = sequence.pad_sequences(combined, maxlen=self.maxlen)
            return w2indx, w2vec, combined
        else:
            print("模型或数据导入失败")

    def wcTrain(self, words):
        '''
        训练word2vec模型，
        :param words:    经过结巴分词后的所有文本内容
        :return:        每个词语的索引，词向量，以及每个句子所对应的词语索引
        '''
        # 训练 word2vec 模型
        model = Word2Vec(size=self.vocab_dim, min_count=self.n_exposures, window=self.window_size, workers=self.cpu_count, iter=self.n_iterations)
        model.build_vocab(words)
        model.train(words)
        # 保存训练好的 word2vec 模型
        model.save(os.path.join(config.BASH_PATH, 'lstm_data', 'Word2vec_model.pkl'))
        # 利用训练好的word2vec模型，和结巴分词后的所有词构建词典
        index_dict, word_vectors, combined = self.buildDic(model=model, words=words)
        return index_dict, word_vectors, combined

    def getData(self, indexDict, wordVectors, combined, y):
        '''
        lstm 初始化，获取相关数据
        :param indexDict:   indexDict为字典，（词语-索引）
        :param wordVectors:  wordVectors为字典，（词语-向量）
        :param combined:    文章中每个句子对应的词语索引 （索引）
        :param y:           每个文章对应的标签 （样本标签）
        :return:
        '''
        n_symbols = len(indexDict) + 1
        embedding_weights = np.zeros((n_symbols, self.vocab_dim))
        for word, index in indexDict.items():
            embedding_weights[index, :] = wordVectors[word]
        train_x, text_x, train_y, test_y = train_test_split(combined, y, test_size=0.2)
        print("交叉验证后训练的数据类型为%s，数据格式为%s。测试的数据类型为%s，数据格式为%s。" % (str(type(train_x)), str(train_x.shape), str(type(text_x)), str(text_x.shape)))
        return n_symbols, embedding_weights, train_x, text_x, train_y, test_y

    def train_lstm(self, n_symbols, embedding_weights, train_x, text_x, train_y, test_y):
        '''
        keras lstm 训练模型
        :param n_symbols:大或等于0的整数，字典长度，即输入数据最大下标+1
        :param embedding_weights:初始化权值
        :param train_x:训练集x
        :param text_x:测试集x
        :param train_y:训练集y
        :param test_y:测试集y
        :return:
        '''
        model = Sequential()
        model.add(Embedding(output_dim=self.vocab_dim, input_dim=n_symbols, input_length=self.input_length))
        model.add(Dropout(0.2))
        model.add(LSTM(self.vocab_dim))
        model.add(Dense(units=200, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(units=1, activation='sigmoid'))
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(train_x, train_y, batch_size=self.batch_size, nb_epoch=self.n_epoch, verbose=1, validation_data=(text_x, test_y))
        score = model.evaluate(text_x, test_y, batch_size=self.batch_size)
        yaml_string = model.to_yaml()
        with open(os.path.join(config.BASH_PATH, 'lstm_data', 'lstm.yml'), 'w') as outfile:
            outfile.write(yaml.dump(yaml_string, default_flow_style=True))
        model.save_weights(os.path.join(config.BASH_PATH, 'lstm_data', 'lstm.h5'))
        print('Test score:', score)

    # 训练lstm
    def train(self):
        '''
        训练 LSTM 模型
        :return:
        '''
        print("开始导入数据...")
        data, label = self.loadFile()
        print("开始进行分词...")
        jieba_data = self.tokenizer(data)
        print("开始训练word2vec...")
        index_dict, word_vectors, combined = self.wcTrain(jieba_data)
        print("lstm训练模型初始化...")
        n_symbols, embedding_weights, train_x, text_x, train_y, test_y = self.getData(index_dict, word_vectors, combined, label)
        print("开始训练lstm...")
        self.train_lstm(n_symbols, embedding_weights, train_x, text_x, train_y, test_y)

    def parseStr(self, string):
        '''
        解析字符串
        :param string: 待预测的字符串
        :return:
        '''
        # 结巴分词之后的数据为list
        words = jieba.lcut(string)
        # 将 list 转换为 数组
        words = np.array(words).reshape(1, -1)
        # 加载word2vec的模型
        model = Word2Vec.load(os.path.join(config.BASH_PATH, 'lstm_data', 'Word2vec_model.pkl'))
        # 通过word2vec模型和分词后的文本提炼文本对应对应索引。
        _, _, combined = self.buildDic(model, words)
        return combined

    def lstmPre(self, string):
        '''
        lstm模型预测
        :param string:待预测的文本
        :return:
        '''

        data = self.parseStr(string)
        data.reshape(1, -1)
        result = self.model.predict_classes(data)
        if result[0][0] == 1:
            str = "该文本是正面的！"
        else:
            str = '该文本是负面的！'
        return str


if __name__ == '__main__':
    lstm = LSTM()
    server = SimpleXMLRPCServer(("192.168.102.38", 8888))
    server.register_instance(lstm)
    print("Listening on port 8888........")
    server.serve_forever()
    # string = "我特别讨厌别人说居然离开银行去保险公司，目光短浅！"
    # string2 = "真是太好用了"
    # lstm.lstmPre(string)
    # lstm.lstmPre(string2)
    pass
