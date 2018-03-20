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


def testJieba():
    string = '我是中国人'
    a = jieba.cut(string)
    print('结巴分词后的数据' + str(a))
    print('结巴分词后的数据类型' + str(type(a)))
    l = list(a)
    print(type(l))
    print(l)


def addNum(addNum, num):
    return addNum + num


def testNumpy():
    np_arange = np.arange(1, 10)
    # a = [addNum(10, a) for a in np_arange]
    # concatenate = np.concatenate(a)

    cw = lambda a: addNum(10, a)
    apply = cw(np_arange)
    print(type(apply))

if __name__ == '__main__':
    testNumpy()
    pass
