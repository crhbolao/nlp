#!/usr/bin/python
# Created with pycharm.
# File Name: ffileUtils
# User: sssd
# Date: 2018/4/10 9:28
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:   用来操作 excel 的表格

import pandas as pd
import os
import time

import re

from sentiment_analysis.util.ShowProcess import ShowProcess

NEG = r'C:\Users\sssd\Desktop\data\neg.txt'
POS = r'C:\Users\sssd\Desktop\data\pos.txt'
MID = r'C:\Users\sssd\Desktop\data\mid.txt'
r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。»?★、…【】 \r\n《》？“”‘’！[\\]^_`{|}~]+'

year = time.localtime().tm_year
mon = time.localtime().tm_mon
day = time.localtime().tm_mday
hour = time.localtime().tm_hour
min = time.localtime().tm_min
sec = time.localtime().tm_sec
nowtime = str(year) + str(mon) + str(day) + str(hour) + str(min) + str(sec)


def readExcel(filePath):
    '''
    给定文件目录的地址进行读取 excel 文件
    :param filePath:  文件目录路径
    :return:
    '''
    pathDir = os.listdir(filePath)
    for dir in pathDir:
        allPath = os.path.join(filePath, dir)
        df = pd.read_excel(allPath, header=0, index=0)
        fileNeg = open(NEG, 'a+')
        fileMid = open(MID, 'a+')
        filePos = open(POS, 'a+')
        for indexs in df.index:
            try:
                print("开始读取%s文件的第%d行" % (str(dir), indexs))
                num = df.iat[indexs, -1]
                if num == 0:
                    file = fileNeg
                elif num == 1:
                    file = fileMid
                else:
                    file = filePos
                tempSeries = df.iloc[indexs, 0:-1]
                tempStr = re.sub(r1, "", tempSeries.values[0].strip())
                file.write(tempStr + '\n')
            except:
                continue
        fileNeg.close()
        fileMid.close()
        filePos.close()


def fileRemove(DirPath):
    '''
    文件去重操作
    :param DirPath:   要去重文件目录
    :return:
    '''
    listdir = os.listdir(DirPath)
    for dir in listdir:
        print("开始处理%s文件" % str(dir))
        allPath = os.path.join(DirPath, dir)

        # 读取文件并去重
        file = open(allPath)
        fileLines = file.readlines()
        uniqueLines = []
        process_bar = ShowProcess(fileLines.__len__())
        for line in fileLines:
            process_bar.show_process()
            if line not in uniqueLines:
                uniqueLines.append(line.replace("\r", "").strip())
        process_bar.close('%s文件去重操作处理完毕' % str(dir))

        # 创建新的文件
        newName = allPath[:allPath.find('.')]
        postfix = allPath[allPath.find('.'):]
        newfile = open(newName + nowtime + postfix, 'a')
        print("保存新的文件路径为：%s" % str(newName + nowtime + postfix))

        # 保存处理过的文件内容
        ShowProcess(uniqueLines.__len__())
        for line in uniqueLines:
            process_bar.show_process()
            newfile.write(line + '\n')
        process_bar.close('新的%s文件保存完毕' % str(newName + nowtime + postfix))
        newfile.close()


if __name__ == '__main__':
    # # 读取excel 文件并按正中负进行分类文件
    # filePath = r'C:\Users\sssd\Desktop\情感帖子'
    # readExcel(filePath)

    dirPath = r'C:\Users\sssd\Desktop\data'
    fileRemove(dirPath)
    pass
