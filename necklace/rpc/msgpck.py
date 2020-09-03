"""RPC Msg Packer"""

import msgpack

try:
    import cPickle as pickle
except:
    import pickle


'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import msgpack

# Define data
data = {'a list': [1, 42, 3.141, 1337, 'help'],
        'a string': 'bla',
        'another dict': {'foo': 'bar',
                         'key': 'value',
                         'the answer': 42}}

# Write msgpack file
with open('data.msgpack', 'w') as outfile:
    msgpack.pack(data, outfile)

# Read msgpack file
with open('data.msgpack') as data_file:
    # data_loaded = json.load(data_file)
    data_loaded = msgpack.unpack(data_file)

print(data == data_loaded)
'''

# https://msgpack.org/
'''
>>> import msgpack
>>> msgpack.packb([1, 2, 3], use_bin_type=True)
'\x93\x01\x02\x03'
>>> msgpack.unpackb(_, raw=False)
[1, 2, 3]

'''

'''
import datetime
import msgpack

useful_dict = {
    "id": 1,
    "created": datetime.datetime.now(),
}

def decode_datetime(obj):
    if b'__datetime__' in obj:
        obj = datetime.datetime.strptime(obj["as_str"], "%Y%m%dT%H:%M:%S.%f")
    return obj

def encode_datetime(obj):
    if isinstance(obj, datetime.datetime):
        return {'__datetime__': True, 'as_str': obj.strftime("%Y%m%dT%H:%M:%S.%f")}
    return obj


packed_dict = msgpack.packb(useful_dict, default=encode_datetime, use_bin_type=True)
this_dict_again = msgpack.unpackb(packed_dict, object_hook=decode_datetime, raw=False)
'''


"""
def packb(msg, *args, **kwargs):
    o = msgpack.packb(msg, use_bin_type=True)
    return o


def unpackb(msg, *args, **kwargs):
    o = msgpack.unpackb(msg, raw=False)
    return o
"""



#https://github.com/lebedov/msgpack-numpy


import msgpack_numpy as m
import numpy as np


'''
x = np.random.rand(5)
x_enc = msgpack.packb(x, default=m.encode)
x_rec = msgpack.unpackb(x_enc, object_hook=m.decode)
'''


def packb(msg, *args, **kwargs):
    o = msgpack.packb(msg, default=m.encode, use_bin_type=True)
    return o


def unpackb(msg, *args, **kwargs):
    o = msgpack.unpackb(msg, object_hook=m.decode, raw=False)
    return o

# use pickle

def pkpackb(msg, *args, **kwargs):
    msg = pickle.dumps(msg)
    o = msgpack.packb(msg, default=m.encode, use_bin_type=True)
    return o


def unpkpackb(msg, *args, **kwargs):
    o = msgpack.unpackb(msg, object_hook=m.decode, raw=False)
    o = pickle.loads(o)
    return o


def pkl_dumps(o, *args, **kwargs):
    s = pickle.dumps(o)
    return s


def pkl_loads(s, *args, **kwargs):
    o = pickle.loads(s)
    return o
