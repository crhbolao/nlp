# -!- coding: utf-8 -!-
# !/usr/bin/python
# Created with pycharm.
# File Name: RelationAnalysis 
# User: bolao
# Date: 2018/9/6 10:31
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:  主要是用来分析文本之间的关系

import os, sys
import jieba, codecs, math
import jieba.posseg as pseg
import relation_analysis.Config as conf


class RelationAnalysis:
    names = {}
    relationShips = {}
    lineNames = {}
    file_prePath = os.path.join(conf.Base_Path)

    def loadData(self):
        '''
        加载对应的数据进行分析
        :return:
        '''
        # jieba加载词典
        jieba.load_userdict(os.path.join(self.file_prePath, 'resources', 'dict.txt'))
        # 读取本地待分析的文件
        f = open(os.path.join(self.file_prePath, 'resources', 'to_train.txt'), 'r', encoding='utf-8')
        lineNum = 0
        # 对待分析的文件进行遍历
        for line in f.readlines():
            # 结巴分词
            poss = pseg.cut(line)
            # 字典添加数组
            self.lineNames[lineNum] = []
            for w in poss:
                # 如果名字的属性不是名字
                if w.flag != 'nr' or len(w.word) < 2:
                    continue
                # 同一行的名词添加到同一个key中
                self.lineNames[lineNum].append(w.word)
                # 用来记录名字，没出现过的名字添加并设置为0，并记录到relationShips中用来统计其和别的名字的关系
                if self.names.get(w.word) is None:
                    self.names[w.word] = 0
                    self.relationShips[w.word] = {}
                # 名字出现过一次加一。
                self.names[w.word] += 1
            lineNum += 1

        # 主要统计关系，如果一行中名字挨着出现则默认两个名字之间有关系，为其两之间relationShips加一。
        for lineNum in self.lineNames:
            for name1 in self.lineNames[lineNum]:
                for name2 in self.lineNames[lineNum]:
                    if name1 == name2:
                        continue
                    if self.relationShips[name1].get(name2) is None:
                        self.relationShips[name1][name2] = 1
                    else:
                        self.relationShips[name1][name2] = self.relationShips[name1][name2] + 1

    def outData(self):
        with codecs.open(os.path.join(self.file_prePath, 'resources', 'node.txt'), 'w', 'gbk') as f:
            f.write("ID Label Weight\r\n")
            # 遍历器名字以及名字出现的次数
            for name, times in self.names.items():
                f.write(name + " " + name + " " + str(times) + "\r\n")

        with codecs.open(os.path.join(self.file_prePath, 'resources', 'edge.txt'), 'w', 'gbk') as f:
            f.write("Source Target Weight\r\n")
            # 遍历两个名字之间的关系，并记录其两之间的权重。
            for name, edges in self.relationShips.items():
                for v, w in edges.items():
                    if w > 3:
                        f.write(name + " " + v + " " + str(w) + "\r\n")


if __name__ == '__main__':
    analysis = RelationAnalysis()
    analysis.loadData()
    analysis.outData()
    pass
