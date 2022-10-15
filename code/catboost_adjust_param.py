
import pandas as pd
from sklearn.model_selection import train_test_split
from IPython.display import display 

import datetime,json
import numpy as np
import pandas as pd
# import AD_YQS.code.catboost_adjust_param as cb 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score,roc_auc_score,accuracy_score




def load_data(path, train_name):
    train = pd.read_csv(path + train_name)
    X_train = train.drop(["label"], axis=1)
    y_train = train[["label"]]
    return X_train, y_train


path = "/home/lijing/data/covid_19/AD_YQS/code/data/"
train_name = "maccs-H1N1_stand44_edge_balance_train.csv"
test_name = "maccs-H1N1_stand44_edge_balance_test.csv"
X_train, y_train = load_data(path, train_name)
X_validation, y_validation = load_data(path, test_name)

from catboost import CatBoostClassifier
 
import pandas as pd
# categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
 
model = CatBoostClassifier(iterations=100, depth=5,cat_features=X_train,learning_rate=0.5, loss_function='Logloss', logging_level='Verbose')
 
model.fit(X_train,y_train,eval_set=(X_validation, y_validation),plot=True)

import matplotlib.pyplot as plt
 
fea_ = model.feature_importances_
 
fea_name = model.feature_names_

plt.figure(figsize=(10, 10))
 
plt.barh(fea_name,fea_,height =0.5)
# plt.savefig("/home/lijing/data/covid_19/AD_YQS/code/cst.png", dpi=100)
# plt.show()




