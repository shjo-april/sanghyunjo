# Copyright (C) 2024 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import json

def read_json(filepath, encoding=None):
    return json.load(open(filepath, 'r', encoding=encoding))

def write_json(filepath, data, encoding=None):
    json.dump(data, open(filepath, 'w', encoding=encoding), indent='\t', ensure_ascii=False)