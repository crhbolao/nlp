#!/usr/bin/python
# Created with pycharm.
# File Name: filePre
# User: sssd
# Date: 2018/4/23 12:04
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:

import re
import numpy as np

def weibocomplie(text):
    users = re.findall(r'@*:[\u4e00-\u9fa5a-zA-Z0-9_-]{1,}', text)
    result2 = re.findall(r'^[\u4e00-\u9fa5a-zA-Z0-9_-]{1,}', text)
    print(users)
    print(result2)




if __name__ == '__main__':
    a = []
    a.append(1)
    a.append(2)
    print(type(a))
    b = np.array(a)
    print(type(b))
    pass
