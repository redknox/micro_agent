#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# create date 2024/6/20

""" 模块说明。"""

__author__ = 'Haifeng Kong'

import sys


class SampleClass(object):
    """类介绍。
    
    Attributes:
        id: 主键。
        name: 对象姓名。
    """

    def __init__(self, id=0):
        """初始化对象id。"""
        self.id = id


def sampleFunction(id, name):
    """一个函数格式样本。
    
    这是一个编写函数注释的样本。
    
    Args:
        id:对象的id号。
        name:对象的姓名。
    
    Returns:
        布尔型的创建成功标志。
        
    Raises:
        TypeError: id传入的类型不是正整数。
    """
    pass


# main()函数是入口函数。
# 如果没有建立单元测试，那么在main()函数里
# 写对模块的测试。

def main():
    # TODO(konghaifeng@cmcm.com): 写测试。
    pass


if __name__ == '__main__':
    main()
