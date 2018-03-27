#!/usr/bin/python
# Created with pycharm.
# File Name: client
# User: sssd
# Date: 2018/3/27 13:40
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description:   python rpc 客户端，类似于 Java 中的dubbo

import xmlrpc.client

if __name__ == '__main__':
    s = xmlrpc.client.ServerProxy('http://192.168.102.38:8888')
    string = "我特别讨厌别人说居然离开银行去保险公司，目光短浅！"
    pre = s.lstmPre(string)
    print(pre)
    pass
