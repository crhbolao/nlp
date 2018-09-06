# -!- coding: utf-8 -!-
# !/usr/bin/python
# Created with pycharm.
# File Name: LstmTrain
# User: sssd
# Date: 2018/4/24 16:23
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:   训练 lstm 模型


import os
import sentiment_analysis.config as config
import jieba
import numpy as np
from keras.preprocessing import sequence
from gensim.corpora.dictionary import Dictionary
from gensim.models import KeyedVectors
import multiprocessing
import pandas as pd
from gensim.models.word2vec import Word2Vec
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml
from sklearn.model_selection import train_test_split
import yaml
from keras.utils import np_utils, generic_utils


class LstmTrain:

    def __init__(self):
        '''
        lstm训练模型初始化
        '''
        # word2vec 的参数配置
        self.vocab_dim = 100
        self.n_exposures = 1  # 统计字数出现的频率
        self.window_size = 7
        self.cpu_count = multiprocessing.cpu_count()
        self.n_iterations = 1  # ideally more..

        # LSTM 相关参数
        self.maxlen = 50  # 输入数据的维度
        self.vocab_dim = 30  # 权值向量的维度
        self.input_length = 50
        self.batch_size = 50  # 批数据大小
        self.n_epoch = 15  # 设置的数据批次

    def loadFile(self, dictPath):
        '''
        加载训练数据
        :return:    合并后的数据，以及数据对应的标签
        '''
        negData = dictPath.get('negData')
        posData = dictPath.get('posData')

        neg = np.loadtxt(negData, encoding='UTF-8', dtype=str)
        pos = np.loadtxt(posData, encoding='UTF-8', dtype=str)
        data = np.concatenate((pos, neg))
        label = np.concatenate((np.ones(len(pos), dtype=int) * 1, np.zeros(len(neg), dtype=int)))
        # data = np.concatenate((pos, mid, neg))
        # label = np.concatenate((np.ones(len(pos), dtype=int) * 1, np.zeros(len(mid), dtype=int), np.ones(len(neg), dtype=int) * (-1)))
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
            dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
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
        model.train(words, epochs=model.iter, total_examples=model.corpus_count)
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
        train_x, text_x, train_y, test_y = train_test_split(combined, y, test_size=0.3)
        print("交叉验证后训练的数据类型为%s，数据格式为%s。测试的数据类型为%s，数据格式为%s。" % (str(type(train_x)), str(train_x.shape), str(type(text_x)), str(text_x.shape)))
        return n_symbols, embedding_weights, train_x, text_x, train_y, test_y

    def train_lstm(self, n_symbols, train_x, text_x, train_y, test_y):
        '''
        keras lstm 训练模型
        :param n_symbols:大或等于0的整数，字典长度，即输入数据最大下标+1
        :param train_x:训练集x
        :param text_x:测试集x
        :param train_y:训练集y
        :param test_y:测试集y
        :return:
        '''
        model = Sequential()

        # 后端是 Thean 的文本预测
        # model.add(Embedding(output_dim=vocab_dim,
        #                     input_dim=n_symbols,
        #                     mask_zero=True,
        #                     weights=[embedding_weights],
        #                     input_length=input_length))
        # model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
        # model.add(Dropout(0.5))
        # model.add(Dense(1))
        # model.add(Activation('sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=n_epoch, verbose=1, validation_data=(text_x, test_y), show_accuracy=True)
        # score = model.evaluate(text_x, test_y, batch_size=batch_size)

        # 后端是tensorflow
        train_y = np_utils.to_categorical(train_y, num_classes=2)
        test_y = np_utils.to_categorical(test_y, num_classes=2)
        model.add(Embedding(output_dim=self.vocab_dim, input_dim=n_symbols, input_length=self.input_length))
        model.add(Dropout(0.2))
        model.add(LSTM(self.vocab_dim))
        model.add(Dense(units=180, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=2, activation='softmax'))
        model.add(Activation('softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(train_x, train_y, batch_size=self.batch_size, nb_epoch=self.n_epoch, verbose=1, validation_data=(text_x, test_y))
        score = model.evaluate(text_x, test_y, batch_size=self.batch_size)

        yaml_string = model.to_yaml()
        with open(os.path.join(config.BASH_PATH, 'lstm_data', 'lstm.yml'), 'w') as outfile:
            outfile.write(yaml.dump(yaml_string, default_flow_style=True))
        model.save_weights(os.path.join(config.BASH_PATH, 'lstm_data', 'lstm.h5'))
        print('Test score:', score)

    def train(self, dictPaths):
        '''
        训练 LSTM 模型
        :return:
        '''
        print("开始导入数据...")
        data, label = self.loadFile(dictPaths)
        print("开始进行分词...")
        jieba_data = self.tokenizer(data)
        print("开始训练word2vec...")
        index_dict, word_vectors, combined = self.wcTrain(jieba_data)
        print("lstm训练模型初始化...")
        n_symbols, embedding_weights, train_x, text_x, train_y, test_y = self.getData(index_dict, word_vectors, combined, label)
        print("开始训练lstm...")
        self.train_lstm(n_symbols, train_x, text_x, train_y, test_y)


if __name__ == '__main__':
    lstm_train = LstmTrain()
    dictPath = {"midData": r'C:\Users\sssd\Desktop\lstm模型文件\fileData\newmid.txt',
                "negData": r'C:\Users\sssd\Desktop\lstm模型文件\fileData\newneg.txt',
                "posData": r'C:\Users\sssd\Desktop\lstm模型文件\fileData\newpos.txt'}
    lstm_train.train(dictPath)

    pass
