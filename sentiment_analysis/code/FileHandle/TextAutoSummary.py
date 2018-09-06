# -!- coding: utf-8 -!-
# !/usr/bin/python
# Created with pycharm.
# File Name: TextAutoSummary
# User: sssd
# Date: 2018/4/23 13:52
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:          自动提取文章摘要


import jieba
import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer


class TextAutoSummary:

    def __init__(self):
        self.stopPath = r'C:\Users\sssd\Desktop\stopWords.txt'

    def cut_sentence(self, sentence):
        """
        分句
        :param sentence:
        :return:
        """
        delimiters = frozenset(u'。！？')
        buf = []
        for ch in sentence:
            buf.append(ch)
            if delimiters.__contains__(ch):
                yield ''.join(buf)
                buf = []
        if buf:
            yield ''.join(buf)

    def load_stopwords(self):
        '''
        记载停留词
        :return:
        '''
        # 打开停留词的文件路径
        with open(self.stopPath, encoding='UTF-8') as f:
            stopwords = list(map(lambda x: x.strip(), f.readlines()))
        stopwords.extend([' ', '\t', '\n'])
        # 将list转换为set。
        return frozenset(stopwords)

    def cut_words(self, sentence):
        """
        分词
        :param sentence:
        :return:
        """
        stopwords = self.load_stopwords()
        return filter(lambda x: not stopwords.__contains__(x), jieba.cut(sentence))

    def get_abstract(self, content, size=3):
        """
        利用textrank提取摘要
        :param content:
        :param size:
        :return:
        """
        docs = list(self.cut_sentence(content))
        tfidf_model = TfidfVectorizer(tokenizer=jieba.cut, stop_words=self.load_stopwords())
        tfidf_matrix = tfidf_model.fit_transform(docs)
        normalized_matrix = TfidfTransformer().fit_transform(tfidf_matrix)
        similarity = nx.from_scipy_sparse_matrix(normalized_matrix * normalized_matrix.T)
        scores = nx.pagerank(similarity)
        tops = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        size = min(size, len(docs))
        indices = list(map(lambda x: x[0], tops))[:size]
        return list(map(lambda idx: docs[idx], indices))


if __name__ == '__main__':
    s = '电动车驾驶人闯红灯被判全责 浙江金华霸气交警走红 2018-04-15 中国新闻网 原标题：电动车驾驶人闯红灯被判全责 浙江金华霸气交警走红 中新网金华4月15日电（记者 奚金燕 通讯员 钱玉蓉 张嘉欣）“交警同志，这种情况下对方一点责任都没有吗？！”近日，一则视频在网络上走红，视频中面对交警的定责， ​ ...展开全文c'
    text_auto_summary = TextAutoSummary()
    for i in text_auto_summary.get_abstract(s, 3):
        print(i)
    pass
