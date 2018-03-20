#!/usr/bin/python
# -*- coding: utf-8 -*-
# Created with pycharm.
# User: sssd
# Date: 2018/3/20 15:52
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:

import os

BASH_PATH = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    print(BASH_PATH)
    print(os.path.join(BASH_PATH, "data"))
    pass
