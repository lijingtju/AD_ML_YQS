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

# plot feature importance manually
from numpy import loadtxt
from xgboost import XGBClassifier
from matplotlib import pyplot

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# feature importance
print("-----")
print(model.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.savefig("/home/lijing/data/covid_19/AD_YQS/code/test.jpg")

from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
model = XGBClassifier()
model.fit(X_train, y_train)
# plot feature importance
plot_importance(model)
pyplot.savefig("/home/lijing/data/covid_19/AD_YQS/code/test.jpg")

