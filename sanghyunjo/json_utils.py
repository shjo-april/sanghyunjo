# Copyright (C) 2024 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import json

from .cv_utils import deprecated

def jsread(filepath, encoding='utf-8'):
    return json.load(open(filepath, 'r', encoding=encoding))

def jswrite(filepath, data, encoding='utf-8'):
    json.dump(data, open(filepath, 'w', encoding=encoding), indent='\t', ensure_ascii=False)

""" Deprecated aliases with warning """
@deprecated("jsread")
def read_json(filepath, encoding=None):
    return jsread(filepath, encoding)

@deprecated("jswrite")
def write_json(filepath, data, encoding=None):
    return jswrite(filepath, data, encoding)
