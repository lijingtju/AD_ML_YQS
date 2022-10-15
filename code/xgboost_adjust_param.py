import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional     scklearn functions
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import warnings
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
# %matplotlib inline
from matplotlib.pylab import rcParams
def load_data(path, train_name):
    train = pd.read_csv(path + train_name)
    X_train = train.drop(["label"], axis=1)
    y_train = train[["label"]]
    return X_train, y_train


path = "/home/lijing/data/covid_19/AD_YQS/code/data/"
train_name = "maccs-H1N1_stand44_edge_balance_train.csv"
test_name = "maccs-H1N1_stand44_edge_balance_test.csv"
X_train, y_train = load_data(path, train_name)
X_test, y_test = load_data(path, test_name) 

params = {
    'booster':'gbtree',
    'objective':'multi:softmax',   # 多分类问题
    'num_class':10,  # 类别数，与multi softmax并用
    'gamma':0.1,    # 用于控制是否后剪枝的参数，越大越保守，一般0.1 0.2的样子
    'max_depth':12,  # 构建树的深度，越大越容易过拟合
    'lambda':2,  # 控制模型复杂度的权重值的L2 正则化项参数，参数越大，模型越不容易过拟合
    'subsample':0.7, # 随机采样训练样本
    'colsample_bytree':3,# 这个参数默认为1，是每个叶子里面h的和至少是多少
    # 对于正负样本不均衡时的0-1分类而言，假设h在0.01附近，min_child_weight为1
    #意味着叶子节点中最少需要包含100个样本。这个参数非常影响结果，
    # 控制叶子节点中二阶导的和的最小值，该参数值越小，越容易过拟合
    'silent':0,  # 设置成1 则没有运行信息输入，最好是设置成0
    'eta':0.007,  # 如同学习率
    'seed':1000,
    'nthread':7,  #CPU线程数
    #'eval_metric':'auc'
}

# #max_depth和min_child_weight参数调优
# # max_depth和min_child_weight参数对最终结果有很大的影响。max_depth通常在3-10之间，min_child_weight。采用栅格搜索（grid search），我们先大范围地粗略参数，然后再小范围的微调。
# # 网格搜索scoring = 'roc_auc' 只支持二分类，多分类需要修改scoring（默认支持多分类）

param_test1 = {
'max_depth':[i for i in range(3,10,2)],
'min_child_weight':[i for i in range(1,6,2)]
}
# from sklearn import svm, grid_search, datasets
# from sklearn import grid_search
gsearch = GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=140, max_depth=5,
min_child_weight=1,
gamma=0,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test1,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch.fit(X_train,y_train)
print('max_depth_min_child_weight')
print(gsearch.best_score_, gsearch.best_params_)

# gamma参数调优
# 　　在已经调整好其他参数的基础上，我们可以进行gamma参数的调优了。Gamma参数取值范围很大，这里我们设置为5，其实你也可以取更精确的gamma值。


param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch = GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=140,
max_depth=4,
min_child_weight=5,
gamma=0,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test3,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch.fit(X_train,y_train)
print('gamma')
print(gsearch.best_score_, gsearch.best_params_)

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch = GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=177,
max_depth=4,
min_child_weight=5,
gamma=0.4,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test4,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch.fit(X_train,y_train)
print('subsample_colsample_bytree------------------')
print(gsearch.best_score_, gsearch.best_params_)

param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

gsearch = GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=177,
max_depth=4,
min_child_weight=5,
gamma=0.4,
subsample=0.9,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test6,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch.fit(X_train,y_train)
print('reg_alpha------------------')
print(gsearch.best_score_, gsearch.best_params_)

param_test7 = {
 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch = GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=177,
max_depth=4,
min_child_weight=5,
gamma=0.4,
subsample=0.9,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test7,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch.fit(X_train,y_train)
print('reg_lambda------------------')
print(gsearch.best_score_, gsearch.best_params_)

model = XGBClassifier(
learning_rate =0.1,
n_estimators=177,
max_depth=4,
min_child_weight=5,
gamma=0.4,
subsample=0.9,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27)
model.fit(X_train, y_train)
y_pre = model.predict(X_test)
print("acc:",metrics.accuracy_score(y_test,y_pre))
print("auc:",metrics.roc_auc_score(y_test,y_pre))
