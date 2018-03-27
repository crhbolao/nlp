#!/usr/bin/python
# Created with pycharm.
# File Name: service.py
# User: sssd
# Date: 2018/3/27 13:38
# Version: V1.0
# To change this template use File | Settings | File Templates.
# Description: python rpc 服务端

from xmlrpc.server import SimpleXMLRPCServer
import calendar


class Calendar:
    def addNum(self, a):  # 两个参数，返回一个字符串对象，即year年month的日历
        data = 1.0 + int(a)
        return data


if __name__ == '__main__':
    calendar_object = Calendar()
    server = SimpleXMLRPCServer(("192.168.102.38", 8888))
    server.register_instance(calendar_object)
    print("Listening on port 8888........")
    server.serve_forever()
