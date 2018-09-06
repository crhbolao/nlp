#!/usr/bin/python
# Created with pycharm.
# File Name: FilePreHandle.py
# User: sssd
# Date: 2018/4/8 10:34
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:    文件预处理

import pandas as pd
import numpy as np
import re
from sentiment_analysis.code.FileHandle.TextAutoSummary import TextAutoSummary
from sentiment_analysis.code.FileHandle.FileMatch import FileMatch


class FilePreHandle:

    def __init__(self):
        self.text_auto_summary = TextAutoSummary()
        self.fileMatch = FileMatch()
        self.summary_num = 2

    def init(self, dictArg):
        self.midPath = dictArg.get('midPath')
        self.posPath = dictArg.get("posPath")
        self.negPath = dictArg.get("negPath")

    def readFile(self, filePath=None):
        '''
        读取训练数据的文件
        :return:
        '''
        if filePath is not None:
            neg = []
            pos = []
            mid = []
            read_excel = pd.read_excel(filePath, header=0, index=0)

            for index in read_excel.index:
                print('开始处理第%d行数据.....' % index)
                # 首先获取待分析的内容
                if read_excel.iloc[index, 6] == '新浪微博':
                    res1 = self.fileMatch.matchText(read_excel.iloc[index, 3])
                    try:
                        temp = self.text_auto_summary.get_abstract(res1, self.summary_num)
                    except:
                        temp = res1
                else:
                    try:
                        temp = self.text_auto_summary.get_abstract(read_excel.iloc[index, 3], self.summary_num)
                    except:
                        temp = read_excel.iloc[index, 3]
                if str(temp).strip():
                    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。»?★、…【】 \r\n《》？“”‘’！[\\]^_`{|}~]+'
                    tempStr = re.sub(r1, "", str(temp))
                    if tempStr.strip():
                        if read_excel.iloc[index, 4] == '中性':
                            mid.append(tempStr)
                        elif read_excel.iloc[index, 4] == '正面':
                            pos.append(tempStr)
                        elif read_excel.iloc[index, 4] == '负面':
                            neg.append(tempStr)

            np.savetxt(self.midPath, np.array(mid), fmt="%s", encoding="UTF-8")
            np.savetxt(self.negPath, np.array(neg), fmt="%s", encoding="UTF-8")
            np.savetxt(self.posPath, np.array(pos), fmt="%s", encoding="UTF-8")


if __name__ == '__main__':
    filePath = r'C:\Users\sssd\Desktop\（第二阶段）政府相关数据情感判别20180425.xlsx'
    # filePath = r'C:\Users\sssd\Desktop\test.xlsx'
    paths = {"midPath": r'C:\Users\sssd\Desktop\lstm模型文件\fileData2\mid.txt',
             "negPath": r'C:\Users\sssd\Desktop\lstm模型文件\fileData2\neg.txt',
             "posPath": r'C:\Users\sssd\Desktop\lstm模型文件\fileData2\pos.txt'}
    file_pre_handle = FilePreHandle()
    file_pre_handle.init(paths)
    file_pre_handle.readFile(filePath)
    pass
