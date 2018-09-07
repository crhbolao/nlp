# -!- coding: utf-8 -!-
# !/usr/bin/python
# Created with pycharm.
# File Name: Config.py 
# User: bolao
# Date: 2018/9/6 10:24
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:  用来获取项目的的基本结构

import os

# 主要是用来获取该目录的基本路径
Base_Path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    print(Base_Path)
    print(os.path.join(Base_Path, 'resources'))
    pass
