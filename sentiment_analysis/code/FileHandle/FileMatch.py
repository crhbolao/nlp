# -!- coding: utf-8 -!-
# !/usr/bin/python
# Created with pycharm.
# File Name: FileMatch
# User: sssd
# Date: 2018/4/23 15:36
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:   文本的正则匹配

import re


class FileMatch:

    def matchText(self, text=None):
        '''
        用来匹配新浪微博中的有用内容
        :param text:
        :return:
        '''
        res = ''
        if text is not None:
            r1 = u'//@((?!:).)*:'
            # 首先正则匹配微博评论中的评论x
            tempStr = re.sub(r1, ",", str(text)).split(",")
            for x in tempStr:
                if x.strip():
                    res = x
                    break
        if not res:
            res = text.strip()
        return res


if __name__ == '__main__':
    string = '//@慢慢长大的流浪猫://@风流才子第一人:袭警//@长春蓝视://@淮河之声://@忧国忧民王全杰:英勇的好民警，人民的守护神。'
    string2 = '//@青丘冷月: //@军师的微博:本应如此!对于遵纪守法的民众来说是好事，只有警察敢于执法，才能真正保护人民群众。'
    file_match = FileMatch()
    text = file_match.matchText(string2)
    print(text)
    pass
