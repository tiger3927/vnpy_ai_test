# encoding: UTF-8
import warnings
import time
import sys
import os
# warnings.filterwarnings("ignore")
from pymongo import MongoClient, ASCENDING
import pandas as pd
import numpy as np
from datetime import datetime
import talib
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.model_selection import train_test_split
# LogisticRegression 逻辑回归
from sklearn.linear_model import LogisticRegression
# DecisionTreeClassifier 决策树
from sklearn.tree import DecisionTreeClassifier
# SVC 支持向量分类
from sklearn.svm import SVC
# MLP 神经网络
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
import joblib
from sklearn.model_selection import StratifiedKFold  # 交叉验证
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif

import random
from typing import List
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.setting import SETTINGS
from vnpy_ctastrategy import (
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)
from tiger_tools import load_json
from numba import jit
from sklearn.preprocessing import StandardScaler, minmax_scale
from collections import deque
import pandas
import random
from multiprocessing import Queue, Process

config_path = ".vntrader"
if not os.path.exists(config_path):
    config_path = "../.vntrader"
    if not os.path.exists(config_path):
        config_path = "../../.vntrader"
# print(config_path)
# print(os.listdir(config_path))
mysetting = load_json(config_path + "/vt_setting.json")

SETTINGS["database.name"] = mysetting["database.name"]
SETTINGS["database.database"] = mysetting["database.database"]
SETTINGS["database.host"] = mysetting["database.host"]
SETTINGS["database.port"] = mysetting["database.port"]
SETTINGS["database.user"] = mysetting["database.user"]
SETTINGS["database.password"] = mysetting["database.password"]

database = get_database()


class DataAnalyzerforSklearn(object):
    """
    这个类是为了SVM做归纳分析数据，以未来6个bar的斜率线性回归为判断分类是否正确。
    不是直接分析HLOC，而且用下列分非线性参数（就是和具体点位无关）
    1.Percentage
    2.std
    4.MACD
    5.CCI
    6.ATR
    7. 该bar之前的均线斜率
    8. RSI
    """

    def __init__(self, name, symbol="ETHUSDT", exchange=Exchange.BINANCE, daily_end=None):
        self.name = name
        self.symbol = symbol
        self.exchange = exchange
        if daily_end:
            self.daily_end = daily_end

        self.df = pd.DataFrame()
        self.data_path = "./data/"
        self.startBar = 2
        self.endBar = 12
        self.step = 2
        self.pValue = 0.015

        self.interval = Interval.MINUTE
        self.start = datetime(2022, 1, 1)
        self.end = datetime(2022, 1, 10)
        self.daily_end = datetime(2023, 10, 10, 23, 59, 0).time()

        self.bars_1m = []
        self.bars_5m = []
        self.bars_15m = []
        self.bars_30m = []
        self.bars_1h = []
        self.bars_4h = []
        self.bars_12h = []
        self.bars_1d = []
        self.bg1m = BarGenerator(self.on_bar, 1, self.on_1min_bar)  # 只需要对 bg1m 使用 update_bar 即可，其他自动产生
        self.bg5m = BarGenerator(self.on_bar, 5, self.on_5min_bar)
        self.bg15m = BarGenerator(self.on_bar, 15, self.on_15min_bar)
        self.bg30m = BarGenerator(self.on_bar, 30, self.on_30min_bar)
        self.bg1h = BarGenerator(self.on_bar, 1, self.on_1h_bar, interval=Interval.HOUR)
        self.bg4h = BarGenerator(self.on_bar, 4, self.on_4h_bar, interval=Interval.HOUR)
        self.bg12h = BarGenerator(self.on_bar, 12, self.on_12h_bar, interval=Interval.HOUR)
        self.bg1d = BarGenerator(self.on_bar, 1, self.on_1d_bar, interval=Interval.DAILY, daily_end=self.daily_end)
        self.df_1m: pandas.DataFrame = None
        self.df_5m: pandas.DataFrame = None
        self.df_15m: pandas.DataFrame = None
        self.df_30m: pandas.DataFrame = None
        self.df_1h: pandas.DataFrame = None
        self.df_4h: pandas.DataFrame = None
        self.df_12m: pandas.DataFrame = None
        self.df_1d: pandas.DataFrame = None

    def on_bar(self, bar: BarData):
        pass

    def on_1min_bar(self, bar: BarData):
        self.bars_1m.append(bar)

    def on_5min_bar(self, bar: BarData):
        self.bars_5m.append(bar)

    def on_15min_bar(self, bar: BarData):
        self.bars_15m.append(bar)

    def on_30min_bar(self, bar: BarData):
        self.bars_30m.append(bar)

    def on_1h_bar(self, bar: BarData):
        self.bars_1h.append(bar)

    def on_4h_bar(self, bar: BarData):
        self.bars_4h.append(bar)

    def on_12h_bar(self, bar: BarData):
        self.bars_12h.append(bar)

    def on_1d_bar(self, bar: BarData):
        self.bars_1d.append(bar)

    # -----------------------------------------导入数据-------------------------------------------------
    def load(self, start, end):
        self.start = start
        self.end = end
        t = time.time()
        bars = database.load_bar_data(
            symbol=self.symbol,
            exchange=self.exchange,
            interval=self.interval,
            start=self.start,
            end=self.end
        )
        print("调取数据库，耗时:", round(time.time() - t, 2), " 秒")
        self.load_bars(bars)

    def load_bars(self, bars: []):  # 用于从其他对象移植数据过来，不重复加载数据库
        # 根据window，计算start 和 total
        print("生成 5m 15m 30m 1h 4h 12h 1d ---- bars 数据 ----")
        t = time.time()
        for b in bars:
            self.bg1m.update_bar(b)
            self.bg5m.update_bar(b)
            self.bg15m.update_bar(b)
            self.bg30m.update_bar(b)
            self.bg1h.update_bar(b)
            self.bg4h.update_bar(b)
            self.bg12h.update_bar(b)
            self.bg1d.update_bar(b)
        print("bars多周期生成耗时：", time.time() - t, " 秒")
        t = time.time()

        def bars_pandas(bars: [BarData]):
            data = np.ndarray(shape=(len(bars), 6))
            i = 0
            ds = []
            for b in bars:
                data[i, 1] = b.open_price
                data[i, 2] = b.high_price
                data[i, 3] = b.low_price
                data[i, 4] = b.close_price
                data[i, 5] = b.volume
                ds.append(b.datetime)
                i += 1
            dt_series = pandas.Series(ds)
            print(len(ds))
            df: pandas.DataFrame = pandas.DataFrame(columns=["open", "high", "low", "close", "volume"], index=dt_series)
            df["open"] = data[:, 1]
            df["high"] = data[:, 2]
            df["low"] = data[:, 3]
            df["close"] = data[:, 4]
            df["volume"] = data[:, 5]
            df.index.name = "datetime"
            print(df.head())
            return df

        # pandas
        self.df_1m = bars_pandas(self.bars_1m)
        self.df_5m = bars_pandas(self.bars_5m)
        self.df_15m = bars_pandas(self.bars_15m)
        self.df_30m = bars_pandas(self.bars_30m)
        self.df_1h = bars_pandas(self.bars_1h)
        self.df_4h = bars_pandas(self.bars_4h)
        self.df_12h = bars_pandas(self.bars_12h)
        self.df_1d = bars_pandas(self.bars_1d)
        print("pandas 生成耗时：", time.time() - t, " 秒")

    def csv2df(self, csvname, dataname="csv_data", export2csv=False):
        """读取csv行情数据，输入到Dataframe中"""
        self.df = pd.read_csv(self.data_path + csvname)
        self.df["datetime"] = pd.to_datetime(self.df['datetime'])
        self.df = self.df.reset_index(drop=True)
        path = self.data_path + dataname + ".csv"
        if export2csv == True:
            self.df.to_csv(path, index=True, header=True)
        return self.df

    def df2Barmin(self, inputdf, barmins, crossmin=1, export2csv=False):
        """输入分钟k线dataframe数据，合并多多种数据，例如三分钟/5分钟等，如果开始时间是9点1分，crossmin = 0；如果是9点0分，crossmin为1"""
        dfbarmin = pd.DataFrame()
        highBarMin = 0
        lowBarMin = 0
        openBarMin = 0
        volumeBarmin = 0
        datetime = 0
        for i in range(0, len(inputdf) - 1):
            bar = inputdf.iloc[i, :].to_dict()
            if openBarMin == 0:
                openBarmin = bar["open"]
            if highBarMin == 0:
                highBarMin = bar["high"]
            else:
                highBarMin = max(bar["high"], highBarMin)
            if lowBarMin == 0:
                lowBarMin = bar["low"]
            else:
                lowBarMin = min(bar["low"], lowBarMin)
            closeBarMin = bar["close"]
            datetime = bar["datetime"]
            volumeBarmin += int(bar["volume"])
            # X分钟已经走完
            if not (bar["datetime"].minute + crossmin) % barmins:  # 可以用X整除
                # 生成上一X分钟K线的时间戳
                barMin = {'datetime': datetime, 'high': highBarMin, 'low': lowBarMin, 'open': openBarmin,
                          'close': closeBarMin, 'volume': volumeBarmin}
                dfbarmin = dfbarmin.append(barMin, ignore_index=True)
                highBarMin = 0
                lowBarMin = 0
                openBarMin = 0
                volumeBarmin = 0
        if export2csv == True:
            dfbarmin.to_csv(self.data_path + "bar" + str(barmins) + str(self.collection) + ".csv", index=True,
                            header=True)
        return dfbarmin

    # -----------------------------------------开始计算指标-------------------------------------------------
    def dfcci(self, inputdf, n):
        """调用talib方法计算CCI指标，写入到df并输出"""
        dfcci = inputdf
        dfcci["cci"] = None
        for i in range(n, len(inputdf)):
            df_ne = inputdf.loc[i - n + 1:i, :]
            cci = talib.CCI(np.array(df_ne["high"]), np.array(df_ne["low"]), np.array(df_ne["close"]), n)
            dfcci.loc[i, "cci"] = cci[-1]
        dfcci = dfcci.fillna(0)
        dfcci = dfcci.replace(np.inf, 0)
        # if export2csv == True:
        #     dfcci.to_csv(self.data_path + "dfcci" + str(self.collection) + ".csv", index=True, header=True)
        return dfcci

    def dfatr(self, inputdf, n):
        """调用talib方法计算ATR指标，写入到df并输出"""
        dfatr = inputdf
        for i in range((n + 1), len(inputdf)):
            df_ne = inputdf.loc[i - n:i, :]
            atr = talib.ATR(np.array(df_ne["high"]), np.array(df_ne["low"]), np.array(df_ne["close"]), n)
            dfatr.loc[i, "atr"] = atr[-1]
        dfatr = dfatr.fillna(0)
        dfatr = dfatr.replace(np.inf, 0)
        return dfatr

    def dfrsi(self, inputdf, n):
        """调用talib方法计算ATR指标，写入到df并输出"""
        dfrsi = inputdf
        dfrsi["rsi"] = None
        for i in range(n + 1, len(inputdf)):
            df_ne = inputdf.loc[i - n:i, :]
            rsi = talib.RSI(np.array(df_ne["close"]), n)
            dfrsi.loc[i, "rsi"] = rsi[-1]
        dfrsi = dfrsi.fillna(0)
        dfrsi = dfrsi.replace(np.inf, 0)
        return dfrsi

    def Percentage(self, inputdf):
        """调用talib方法计算CCI指标，写入到df并输出"""
        dfPercentage = inputdf
        # dfPercentage["Percentage"] = None
        for i in range(1, len(inputdf)):
            # if dfPercentage.loc[i,"close"]>dfPercentage.loc[i,"open"]:
            #     percentage = ((dfPercentage.loc[i,"high"] - dfPercentage.loc[i-1,"close"])/ dfPercentage.loc[i-1,"close"])*100
            # else:
            #     percentage = (( dfPercentage.loc[i,"low"] - dfPercentage.loc[i-1,"close"] )/ dfPercentage.loc[i-1,"close"])*100
            if dfPercentage.loc[i - 1, "close"] == 0.0:
                percentage = 0
            else:
                percentage = ((dfPercentage.loc[i, "close"] - dfPercentage.loc[i - 1, "close"]) / dfPercentage.loc[
                    i - 1, "close"]) * 100.0
            dfPercentage.loc[i, "Perentage"] = percentage
        dfPercentage = dfPercentage.fillna(0)
        dfPercentage = dfPercentage.replace(np.inf, 0)
        return dfPercentage

    def dfMACD(self, inputdf, n):
        """调用talib方法计算MACD指标，写入到df并输出"""
        dfMACD = inputdf
        for i in range(n, len(inputdf)):
            df_ne = inputdf.loc[i - n + 1:i, :]
            macd, signal, hist = talib.MACD(np.array(df_ne["close"]), 12, 26, 9)
            dfMACD.loc[i, "macd"] = macd[-1]
            dfMACD.loc[i, "signal"] = signal[-1]
            dfMACD.loc[i, "hist"] = hist[-1]
        dfMACD = dfMACD.fillna(0)
        dfMACD = dfMACD.replace(np.inf, 0)
        return dfMACD

    def dfSTD(self, inputdf, n):
        """调用talib方法计算MACD指标，写入到df并输出"""
        dfSTD = inputdf
        for i in range(n, len(inputdf)):
            df_ne = inputdf.loc[i - n + 1:i, :]
            std = talib.STDDEV(np.array(df_ne["close"]), n)
            dfSTD.loc[i, "std"] = std[-1]
        dfSTD = dfSTD.fillna(0)
        dfSTD = dfSTD.replace(np.inf, 0)
        return dfSTD

    # -----------------------------------------加入趋势分类-------------------------------------------------
    def addTrend(self, inputdf, trendsetp=6):
        """以未来6个bar的斜率线性回归为判断分类是否正确"""
        dfTrend = inputdf
        for i in range(1, len(dfTrend) - trendsetp - 1):
            histRe = np.array(dfTrend["close"])[i:i + trendsetp]
            xAixs = np.arange(trendsetp) + 1
            res = st.linregress(y=histRe, x=xAixs)
            if res.pvalue < self.pValue + 0.01:
                if res.slope > 0.5:
                    dfTrend.loc[i, "tradeindictor"] = 1
                elif res.slope < -0.5:
                    dfTrend.loc[i, "tradeindictor"] = -1
        dfTrend = dfTrend.fillna(0)
        dfTrend = dfTrend.replace(np.inf, 0)
        return dfTrend


def GirdValuate(X_train, y_train):
    """1）LogisticRegression
    逻辑回归
    2）DecisionTreeClassifier
    决策树
    3）SVC
    支持向量分类
    4）MLP
    神经网络"""
    clf_DT = DecisionTreeClassifier()
    param_grid_DT = {'max_depth': [1, 2, 3, 4, 5, 6]}
    clf_Logit = LogisticRegression()
    param_grid_logit = {'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag']}
    clf_svc = SVC()
    param_grid_svc = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                      'C': [1, 2, 4],
                      'gamma': [0.125, 0.25, 0.5, 1, 2, 4]}
    clf_mlp = MLPClassifier()
    param_grid_mlp = {"hidden_layer_sizes": [(100,), (100, 30)],
                      "solver": ['adam', 'sgd', 'lbfgs'],
                      "max_iter": [20],
                      "verbose": [False]
                      }
    # 打包参数集合
    clf = [clf_DT, clf_Logit, clf_mlp, clf_svc]
    param_grid = [param_grid_DT, param_grid_logit, param_grid_mlp, param_grid_svc]
    kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)  # 将训练/测试数据集划分10个互斥子集，这样方便多进程测试
    # 网格测试
    for i in range(0, 4):
        grid = GridSearchCV(clf[i], param_grid[i], scoring='accuracy', n_jobs=-1, cv=kflod)
        grid.fit(X_train, y_train)
        print(grid.best_params_, ': ', grid.best_score_)


if __name__ == '__main__':
    # 读取数据
    # data_path = "C:\\Users\shui0\OneDrive\Documents\Project\\"
    data_path = "./data/"
    DA = DataAnalyzerforSklearn("ethusdt_1m_1h_4h",symbol= "ETHUSDT", exchange=Exchange.BINANCE)
    # 数据库导入
    DA.load(datetime(2021, 1, 1), datetime(2021, 3, 1))    # df = DA.db2df(db="VnTrader_1Min_Db", collection="rb8888", start=start, end=end)
    df5min = DA.df_5m
    df5min.to_csv(data_path + "df5min" + ".csv", index=True, header=True)
    df5minAdd = DA.addTrend(df5min)
    df5minAdd = DA.dfMACD(df5minAdd, n=34)
    df5minAdd = DA.dfatr(df5minAdd, n=25)
    df5minAdd = DA.dfrsi(df5minAdd, n=35)
    df5minAdd = DA.dfcci(df5minAdd, n=30)
    df5minAdd = DA.dfSTD(df5minAdd, n=30)
    df5minAdd = DA.Percentage(df5minAdd)
    df5minAdd.to_csv(data_path + "df5minAdd" + ".csv", index=True, header=True)
    print("生成数据表，并添加了特征...")
    # 划分测试验证。
    df_test = df5minAdd.loc[60:, :]  # 只从第60个开始分析，因为之前很多是空值
    y = np.array(df_test["tradeindictor"])  # 只保留结果趋势结果，转化为数组
    X = df_test.drop(["tradeindictor", "close", "datetime", "high", "low", "open", "volume"],
                     axis=1).values  # 不是直接分析HLOC，只保留特征值，转化为数组
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # 三七
    print("训练集长度: %s, 测试集长度: %s" % (len(X_train), len(X_test)))

    # 特征工作，可以按照百分比选出最高分特征类，取最优70%，也可以用SelectKBest，指定要几个特征类。
    print(X_train.shape)
    selectPer = SelectPercentile(mutual_info_classif, percentile=70)
    # selectPer = SelectKBest(mutual_info_classif, k=7)
    X_train = selectPer.fit_transform(X_train, y_train)
    print(X_train.shape)
    X_test = selectPer.transform(X_test)
    # 也可以用Fpr选择
    # selectFea=SelectFpr(alpha=0.01)
    # X_train_new = selectFea.fit_transform(X_train, y_train)
    # X_test_new = selectFea.transform(X_test)
    # 这里使用下面模式进行分析，然后利用网格调参
    GirdValuate(X_train, y_train)
    # 使用选取最好的模型，进行测试看看拼接
    # • 模型预测：model.predict()
    # • Accuracy：metrics.accuracy_score()
    # • Presicion：metrics.precision_score()
    # • Recall：metrics.recall_score()

    clf_selected = MLPClassifier(hidden_layer_sizes=(100, 30), max_iter=20, solver='adam')  # 此处填入网格回测最优模型和参数,
    # {'hidden_layer_sizes': (100, 30), 'max_iter': 20, 'solver': 'adam', 'verbose': False} :  0.9897016507648039
    clf_selected.fit(X_train, y_train)
    y_pred = clf_selected.predict(X_test)
    # accuracy
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    print('accuracy:', accuracy)
    # precision
    precision = metrics.precision_score(y_true=y_test, y_pred=y_pred, average="micro")
    print('precision:', precision)
    # recall
    recall = metrics.recall_score(y_true=y_test, y_pred=y_pred, average="micro")
    print('recall:', recall)
    # 实际值和预测值
    print(y_test)
    print(y_pred)
    dfresult = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
    dfresult.to_csv(data_path + "result" + ".csv", index=True, header=True)

    # 模型保存到本地
    joblib.dump(clf_selected, 'clf_selected.m')
    # 模型的恢复
    clf_tmp = joblib.load('clf_selected.m')
