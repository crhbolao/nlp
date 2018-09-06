# -!- coding: utf-8 -!-
# !/usr/bin/python
# Created with pycharm.
# File Name: LstmPre
# User: sssd
# Date: 2018/4/25 14:52
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:    lstm 用来预测语句的正中负

import jieba
import numpy as np
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from gensim.models.word2vec import Word2Vec
import os
import sentiment_analysis.config as config
import yaml
from keras.models import model_from_yaml

jieba.load_userdict(os.path.join(config.BASH_PATH, 'data', 'jiebacutword.txt'))


class LstmPre:

    def __init__(self):

        self.label = ['pos', 'mid', 'neg']
        self.maxlen = 50

        print('加载训练好的数据模型...')
        # 加载 lstm 的网络模型
        with open(os.path.join(config.BASH_PATH, 'lstm_data', 'lstm.yml'), 'r') as f:
            modelStr = yaml.load(f)
        self.model = model_from_yaml(modelStr)

        print("加载训练好的模型参数...")
        # 加载 lstm 模型的权值参数
        self.model.load_weights(os.path.join(config.BASH_PATH, 'lstm_data', 'lstm.h5'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # 加载词典中的负面词。
        self.negWord = []
        negFile = open(os.path.join(config.BASH_PATH, 'data', 'negWord.txt'), encoding='UTF-8')
        for line in negFile.readlines():
            if line.strip():
                self.negWord.append(line.strip())

        # 加载词典中的正面词
        self.posWord = []
        posFile = open(os.path.join(config.BASH_PATH, 'data', 'posWord.txt'), encoding='UTF-8')
        for line in posFile.readlines():
            if line.strip():
                self.posWord.append(line.strip())

        self.allWords = self.posWord + self.negWord

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

    def parseStr(self, string):
        '''
        解析字符串
        :param string: 待预测的字符串
        :return:
        '''
        # 结巴分词之后的数据为list
        words = jieba.lcut(string)
        for index in range(len(words)):
            if words[index] in self.allWords:
                break
            else:
                if (index == len(words) - 1):
                    return []
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
        if len(data) == 0:
            resStr = '该文本是中性的！'
        else:
            data.reshape(1, -1)
            zero_num = np.sum(data == 0)
            result = self.model.predict_classes(data)
            if result[0] == 1:
                resStr = "该文本是正面的！"
            else:
                resStr = '该文本是负面的！'
        return resStr


if __name__ == '__main__':
    string = '法院包庇违法拆迁贪污犯法律何在如违法强拆血告到底（上海浦东康桥秀龙村号）'
    lstm_pre = LstmPre()
    # pre = lstm_pre.lstmPre(string)
    # print(pre)

    file = open(r'C:\Users\sssd\Desktop\lstm模型文件\fileData2\newneg.txt', encoding='UTF-8')
    lines = file.readlines()
    neg_num = 0
    pos_num = 0
    mid_num = 0
    for index, line in enumerate(lines):
        pre = lstm_pre.lstmPre(line)
        print('第%d行的预测结果为：%s' % (index + 1, pre))
        if '负面' in pre:
            neg_num += 1
        elif '正面' in pre:
            pos_num += 1
        else:
            mid_num += 1
    print('预测为负面的个数为：%d，正面的个数为：%d，中性的个数为：%d' % (neg_num, pos_num, mid_num))
    pass
