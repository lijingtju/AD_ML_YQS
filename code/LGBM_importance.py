import pandas as pd
import matplotlib.pylab as plt
import lightgbm as lgb
# read data
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

lgb_clf = lgb.LGBMClassifier(objective='binary',num_leaves=35,max_depth=6,learning_rate=0.05,seed=2018,
        colsample_bytree=0.8,subsample=0.9,n_estimators=20000)
lgb_model = lgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=200)
lgb_predictors = [i for i in X_train.columns]
lgb_feat_imp = pd.Series(lgb_model.feature_importances_, lgb_predictors).sort_values(ascending=False)
print(lgb_feat_imp)
# lgb_feat_imp.to_csv('lgb_feat_imp.csv')
