#!/usr/bin/python
# -*- coding: utf-8 -*-
# Created with pycharm.
# User: sssd
# Date: 2018/3/20 14:04
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:

__author__ = 'sssd'

import jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary


def testJieba():
    string = '我是中国人'
    a = jieba.cut(string)
    print('结巴分词后的数据' + str(a))
    print('结巴分词后的数据类型' + str(type(a)))


def addNum(addNum, num):
    return addNum + num


def testNumpy():
    np_arange = np.arange(1, 10)
    a = [addNum(10, a) for a in np_arange]
    np.concatenate([addNum(10, z) for z in np_arange])

    # cw = lambda a: addNum(10, a)
    # apply = cw(np_arange)
    # print(type(apply))


def testNumoyList():
    df = pd.DataFrame(columns=["a"])
    list1 = [1, 2, 3, 4, 5]

    print(df)
    print("-------------------------------------------")
    print(type(df.iloc(1)))


def testWord2Vec():
    string = '大卫 李髌骨 韧带 撕裂 等待 撕裂'
    line_sent = []
    line_sent.append(string.split())
    model = Word2Vec(line_sent,
                     size=300,
                     window=5,
                     min_count=1,
                     workers=2)
    for i in model.vocab.keys():  # vocab是dict
        print(type(i))
        print(i)
    dict = Dictionary()
    dict.doc2bow(model.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in dict.items()}
    print(type(w2indx))

if __name__ == '__main__':
    testWord2Vec()
    pass
