# encoding: utf-8
"""
@author: Xin Zhang
@contact: meetdevin.zh@outlook.com
@file: my_tools.py
@time: 3/6/19
@desc: 自定义异常类
"""

import glob
import platform


def file_scanf(path, endswith, sub_ratio=1):
    files = glob.glob(path + '/*')
    if platform.system().lower() == 'windows':
        files = [f.replace('\\', '/') for f in files]
    disallowed_file_endings = (".gitignore", ".DS_Store")
    _input_files = files[:int(len(files) * sub_ratio)]
    return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(endswith), _input_files))


class ExceptionPassing(Exception):
    """
    继承自基类 Exception
    """
    def __init__(self, *message, expression=None):
        super(ExceptionPassing, self).__init__(message)
        self.expression = expression
        self.message = str.join('', [str(a) for a in message])