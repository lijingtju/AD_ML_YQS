import pandas as pd
import lightgbm as lgb
from sklearn import metrics

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
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'nthread':4,
          'learning_rate':0.1,
          'num_leaves':30, 
          'max_depth': 5,   
          'subsample': 0.8, 
          'colsample_bytree': 0.8, 
    }
    
data_train = lgb.Dataset(X_train, y_train)
cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',early_stopping_rounds=50,seed=0)
print('best n_estimators:', len(cv_results['auc-mean']))
print('best cv score:', pd.Series(cv_results['auc-mean']).max())

from sklearn.model_selection import GridSearchCV

params_test1={'max_depth': range(3,5,1), 'num_leaves':range(5, 15, 5)}
              
gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=188, max_depth=6, bagging_fraction = 0.8,feature_fraction = 0.8), 
                       param_grid = params_test1, scoring='roc_auc',cv=5,n_jobs=-1)
gsearch1.fit(X_train,y_train)
print(gsearch1.best_score_, gsearch1.best_params_)




params_test2={'max_bin': range(5,256,10), 'min_data_in_leaf':range(1,102,10)}
              
gsearch2 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=188, max_depth=4, num_leaves=10,bagging_fraction = 0.8,feature_fraction = 0.8), 
                       param_grid = params_test2, scoring='roc_auc',cv=5,n_jobs=-1)
gsearch2.fit(X_train,y_train)
print(gsearch2.best_score_, gsearch2.best_params_)


params_test3={'feature_fraction': [0.6,0.7,0.8,0.9,1.0],
              'bagging_fraction': [0.6,0.7,0.8,0.9,1.0],
              'bagging_freq': range(0,81,10)
}
              
gsearch3 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=188, max_depth=4, num_leaves=10,max_bin=15,min_data_in_leaf=51), 
                       param_grid = params_test3, scoring='roc_auc',cv=5,n_jobs=-1)
gsearch3.fit(X_train,y_train)
print(gsearch3.best_score_, gsearch3.best_params_)

params_test4={'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
              'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]
}
              
gsearch4 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=188, max_depth=4, num_leaves=10,max_bin=15,min_data_in_leaf=51,bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 0.8), 
                       param_grid = params_test4, scoring='roc_auc',cv=5,n_jobs=-1)
gsearch4.fit(X_train,y_train)
print(gsearch4.best_score_, gsearch4.best_params_)

params_test5={'min_split_gain':[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
              
gsearch5 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=188, max_depth=4, num_leaves=10,max_bin=15,min_data_in_leaf=51,bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 0.8,
lambda_l1=1e-05,lambda_l2=1e-05), 
                       param_grid = params_test5, scoring='roc_auc',cv=5,n_jobs=-1)
gsearch5.fit(X_train,y_train)
print(gsearch5.best_score_, gsearch5.best_params_)

model=lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.01, n_estimators=1000, max_depth=4, num_leaves=10,max_bin=15,min_data_in_leaf=51,bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 0.8,
lambda_l1=1e-05,lambda_l2=1e-05,min_split_gain=0)
model.fit(X_train,y_train)
y_pre=model.predict(X_test)
print("acc:",metrics.accuracy_score(y_test,y_pre))
print("auc:",metrics.roc_auc_score(y_test,y_pre))

# model=lgb.LGBMClassifier()
# model.fit(X_train,y_train)
# y_pre=model.predict(X_test)
# print("acc:",metrics.accuracy_score(y_test,y_pre))
# print("auc:",metrics.roc_auc_score(y_test,y_pre))








