# -!- coding: utf-8 -!-
# !/usr/bin/python
# Created with pycharm.
# File Name: FileUtils
# User: sssd
# Date: 2018/4/25 8:58
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:     文件处理的工具类

import numpy as np
import os


def FilePreHandle(dirPath=None):
    '''
    用来对某个目录下的文件进行去重
    :param dirPath:   待去重文件的目录
    :return:
    '''
    uniqueFile = []
    if dirPath is not None:
        for fileName in os.listdir(dirPath):
            for line in open(os.path.join(dirPath, fileName), encoding='UTF-8'):
                if line.strip() not in uniqueFile:
                    uniqueFile.append(line.strip())
            np_array = np.array(uniqueFile)
            np.savetxt(os.path.join(dirPath, 'new' + fileName), np_array, fmt="%s", encoding="UTF-8")
            uniqueFile.clear()

if __name__ == '__main__':
    dirPath = r'C:\Users\sssd\Desktop\lstm模型文件\fileData2'
    FilePreHandle(dirPath)
    pass
