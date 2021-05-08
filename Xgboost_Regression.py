# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:06:45 2021

@author: lijiaojiao

xgboost用于回归案例

"""


import xgboost as xgb 
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

#load_boston：波士顿房价数据集
#加载数据集
boston = load_boston()
x,y = boston.data, boston.target

# XGboost 训练过程
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

params = {
    "booster":"gbtree",
    "objective":"reg:squarederror",
    "gamma": 0.1,
    "max_depth" :5,
    "lambda": 3,
    "subsample":0.7,
    "colsample_bytree":0.7,
    "min_child_weight":3,
    "silent":1,
    "eta":0.1,
     "seed":1000,
     "nthread":4,
    
    }

dtrain = xgb.DMatrix(x_train,y_train)
num_rounds = 300

plst = list(params.items())
model = xgb.train(plst,dtrain,num_rounds)

dtest = xgb.DMatrix(x_test,y_test)
ans = model.predict(dtest)

plot_importance(model)
plt.show()







