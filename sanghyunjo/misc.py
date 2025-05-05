# Copyright (C) 2024 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import os
import sys
import time
import glob
import tqdm
import argparse

from typing import Union, Tuple
from joblib import Parallel, delayed

def set_env(params: dict={'CUDA_VISIBLE_DEVICES': '0'}):
    for key in params.keys():
        os.environ[key] = params[key]

def get_env(key: str='CUDA_VISIBLE_DEVICES'):
    return os.environ[key]

def gpus(key='CUDA_VISIBLE_DEVICES'):
    try: os.environ[key]
    except KeyError: os.environ[key] = '0'
    return os.environ[key].split(',')

def cpus(): 
    return os.cpu_count()

def linux():
    return sys.platform == 'linux'

def makedir(path):
    os.makedirs(path, exist_ok=True)
    return path

def basename(path, split_ext=False, remove_ext=False, replace_ext='') -> Union[str, Tuple[str, str]]:
    """
    basename("/some/dir/file.txt")                     # 'file.txt'
    basename("/some/dir/file.txt", split_ext=True)     # ('file', '.txt')
    basename("/some/dir/file.txt", remove_ext=True)    # 'file'
    basename("/some/dir/file.txt", replace_ext='.md')  # 'file.md'
    """
    filename = os.path.basename(path)

    if split_ext or remove_ext or replace_ext:
        file_id, ext = os.path.splitext(filename)

        if split_ext:
            return file_id, ext
        elif remove_ext:
            return file_id
        else:
            return file_id + replace_ext
    
    return filename

def isfile(path): # file or dir
    return os.path.isfile(path) or os.path.isdir(path)

def listdir(dir_path):
    if '*' in dir_path: return glob.glob(dir_path)
    else: return os.listdir(dir_path)

def progress(data, desc=None):
    return tqdm.tqdm(data, desc)

def get_digits(number):
    count = 0
    while number > 0:
        count += 1
        number //= 10
    return count

def parallel(fn, data_list, n_jobs=cpus(), show=False):
    print_dict = {}
    if show: print_dict = {'verbose': 10, 'pre_dispatch': 'all'}

    return Parallel(n_jobs, **print_dict)([delayed(fn)(*data) for data in data_list])

def strfmt(fmt: str='5s', v: str='True'):
    return ('%'+fmt)%v

class Timer:
    def __init__(self):
        self.start = time.time()

    def __enter__(self):
        self.start = time.time()
        return self

    def get(self, unit='ms', as_str=True):
        """Return elapsed time since last call or start."""
        now = time.time()
        elapsed = now - self.start
        self.start = now

        if unit == 's':
            value = elapsed
            suffix = 's'
        else:  # default to milliseconds
            value = elapsed * 1000
            suffix = 'ms'

        return f'{int(value)}{suffix}' if as_str else value

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class Progress:
    def __init__(self, total, desc):
        self.__enter__(total, desc)

    def __enter__(self, total, desc):
        self.pbar = tqdm.tqdm(total=total, desc=desc)
        return self
    
    def update(self):
        self.pbar.update(1)
        
    def __exit__(self, _type, _value, _trackback):
        pass

class Parser:
    def __init__(self, input_dict: dict = None):
        self.parser = argparse.ArgumentParser()
        if input_dict is not None:
            self._add_from_inputs(input_dict)
        self.args = self.parser.parse_args()

        # TODO: automatically convert a string to dictionary
    
    def _add(self, tag, default):
        if isinstance(default, bool):
            option = {"action": "store_false"} if default else {"action": "store_true"}
        elif isinstance(default, list):
            option = {"nargs": "+", "default": default, "type": type(default[0])}
        else:
            option = {"default": default, "type": type(default)}
        self.parser.add_argument(f'--{tag}', **option)

    def _add_from_inputs(self, inputs):
        if isinstance(inputs, list):
            for data in inputs:
                self._add(*data)
        else:
            for tag, default in inputs.items():
                self._add(tag, default)

    def __getattr__(self, name):
        # Prevent recursion by checking internal __dict__
        args = self.__dict__.get('args', None)
        if args is not None:
            return getattr(args, name)
        raise AttributeError(f"'Parser' object has no attribute '{name}'")

    def get(self):
        # Provided for backward compatibility with previous usage
        return self.args