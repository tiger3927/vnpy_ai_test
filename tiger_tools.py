import datetime
import json
import os

import numpy as np
from vnpy.trader.object import BarData, TickData, OrderData, TradeData
import pickle

class CJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, datetime.date):
            return obj.strftime('%Y-%m-%d')
        else:
            return json.JSONEncoder.default(self, obj)

def save_json(o, filename: str) -> bool:
    try:
        sss = json.dumps(o, cls=CJsonEncoder, ensure_ascii=False)
        with open(filename, mode='w') as f:
            f.write(sss)
            f.close()
        return True
    except Exception as e:
        print("save json error:", e)
        return False

def load_json(filename: str):
    s = "{}"
    with open(filename, "r", encoding="utf8") as f:
        s = f.read()
        f.close()
    # print(s)
    return json.loads(s)

def save_data(data, filename: str) -> bool:
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        return False

def load_data(filename: str):
    try:
        with open(filename, "rb") as f:
            o = pickle.load(f)
        return o
    except Exception as e:
        return None

def how_many_digits(v:float):
    i=0
    while v!=round(v,i):
        print(round(v,i))
        i+=1
        if i>32:
            return 32
    return i

class Json_ObjToDictEncoder(json.JSONEncoder):
    def default(self, obj):
        return obj.__dict__

class Json_DictToObjDecoder:
    def __init__(self, d):
        self.__dict__ = d


