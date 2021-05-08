# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:15:33 2021

@author: lijiaojiao
"""

import numpy as np
import pandas as pd
import xgboost as xgb 
#import matplotlib.pyplot as plt
#plt.style.use("ggplot")
#%matplotlib inline

# =============================================================================
# 
# #读数据
# dtrain = xgb.DMatrix("agaricus.txt.train") # XGBoost的专属数据格式，但是也可以用dataframe或者ndarray
# dtest = xgb.DMatrix("agaricus.txt.test")
#  params = {"max_depth":2,"eta":1,"objective":"binary:logistic"}   # 设置XGB的参数，使用字典形式传入
#  num_round = 2 #使用线程数
#  bst = xgb.train(params, dtrain,num_round) #训练
#  preds = bst.predict(dtest) #预测
#  
#  
# =============================================================================
 

#xgboost用于分类案例

from sklearn.datasets import load_iris
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#加载样本数据
iris = load_iris()
x,y = iris.data,iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

#算法参数设置
params = {
    "booster":"gbtree",
    "objective":"multi:softmax",
    "num_class":3,
    "gamma":0.1,
    "max_depth":6,
    "lambda":2,
    "subsample":0.7,
    "colsample_bytree":0.75,
    "min_child_weight":3,
    "silent":0,
    "eta":0.1,
    "seed":1,
    "nthread":4,
    }

plst = list(params.items())

dtrain = xgb.DMatrix(x_train,y_train) #生成相应格式数据集
num_rounds = 500

#xgboost 模型训练
model = xgb.train(plst,dtrain,num_rounds)    

# 对测试集进行预测
dtest = xgb.DMatrix(x_test)
y_pred = model.predict(dtest)

#计算准确率
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy:%.2f%%" %(accuracy*100.0))


#显示重要特征
plot_importance(model)
plt.show()
























 
 
 


