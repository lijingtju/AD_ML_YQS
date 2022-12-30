# *****************************************************************
# Script for different classifiers using Scikit-learn Library *****
# *****************************************************************

from hypopt import GridSearch
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import argparse
import os
import json
from sklearn.externals import joblib
from skmultilearn.adapt import MLkNN
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier



class Model_Development:
    def __init__(self, train_data, valid_data, test_data):
        # score='f1_macro'
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.random_state = 42 
        self.verbose = 2
        # self.score = score

        self.results = dict()
        
        self.output_dir = '/home/lijing/data/covid_19/other_model/redial-2020/redial-2020-notebook-work/report_default_multilabel_lijing'

        if not os.path.isdir(self.output_dir): os.makedirs(self.output_dir)
        
        _, file_name = os.path.split(test_data)
        self.file_name, _ = os.path.splitext(file_name)
        self.result_file = os.path.join(self.output_dir, f"{self.file_name}_results.json")        

    def evaluation_metrics(self, y_true, y_pred, y_prob):
        
        Scores = dict()
        print(y_true.shape, y_pred.shape, y_prob.shape)
        print('+++++++++++++++++++++++++++++++++++++')
        print(y_prob)
        # roc_auc = metrics.roc_auc_score(y_true, y_prob, average = 'samples')
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average = 'samples')
        precision = metrics.precision_score(y_true, y_pred, average= 'samples')
        recall = metrics.recall_score(y_true, y_pred, average = 'samples')

        Scores['ACC'] = acc.tolist()
        Scores['F1_Score'] = f1.tolist()
        Scores['Recall'] = recall.tolist()
        Scores['Precision'] = precision.tolist()

        
        return Scores
        
    def get_data_sets(self):
        
        train_data = np.load(self.train_data)
        valid_data = np.load(self.valid_data)
        test_data = np.load(self.test_data)

        Y_colum = [-3, -2,  -1]
        self.X_train = train_data[:, :-3]
        self.y_train = train_data[:, Y_colum]
        self.X_test = test_data[:, :-3]
        self.y_test = test_data[:, Y_colum]
        self.X_valid = valid_data[:, :-3]
        self.y_valid = valid_data[:, Y_colum]
        print(self.X_train.shape, self.y_train.shape)
        print(self.X_valid.shape, self.y_valid.shape)
        print(self.X_test.shape, self.y_test.shape)

        return True

    
    
    def load_params(self):

        self.params = dict()
        
        self.params['dt'] = {'splitter': ['best'], 'random_state': [self.random_state] }
        self.params['rf'] = {'n_estimators': [100], 'random_state': [self.random_state]}
        self.params['ada'] = {'n_estimators': [50], 'random_state': [self.random_state]}
        self.params['lr'] = {'C': [1.0], 'random_state': [self.random_state]}
        self.params['xgb'] = {'n_estimators': [100], 'random_state': [self.random_state]}
        self.params['out'] = {'code_size': [1.5], 'random_state': [self.random_state]}
        self.params['ovr'] = {'n_jobs': [None]}
        self.params['knb'] = {'n_neighbors': [5]}
        self.params['LGB'] = {
                                'task' : ['predict'],
                                'boosting': ['gbdt' ],
                                'objective': ['root_mean_squared_error'],
                                'num_iterations': [  1500, 2000,5000  ],
                                'learning_rate':[  0.05, 0.005 ],
                                'num_leaves':[ 7, 15, 31  ],
                                'max_depth' :[ 10,15,25],
                                'min_data_in_leaf':[15,25 ],
                                'feature_fraction': [ 0.6, 0.8,  0.9],
                                'bagging_fraction': [  0.6, 0.8 ],
                                'bagging_freq': [   100, 200, 400  ],
                                    
                                }
        self.params['mlknn'] = {'k': [5]}
        self.params['ovo'] = {'n_jobs': [None]}
        self.params['ridge'] = {'alpha': [1.0], 'random_state': [self.random_state]}
        self.params['nr'] = {'metric': ['euclidean']}
        self.params['mnb']= {'alpha': [1.0]}
        self.params['sgd'] = {'alpha': [0.0001], 'random_state': [self.random_state]}
        self.params['mlp'] = {'activation': ['relu'], 'random_state': [self.random_state]}
        self.params['etas'] = {'n_estimators': [100], 'random_state': [self.random_state]}
        self.params['eta'] = {'max_depth': [None], 'random_state': [self.random_state]}
        self.params['pac'] = {'C': [1.0], 'random_state': [self.random_state]}
        self.params['cnb'] = {'alpha': [1.0]}
        self.params['lsvc'] = {'penalty': ['l2'], 'random_state': [self.random_state]}
        self.params['per'] = {'alpha': [0.0001], 'random_state': [self.random_state]}
        self.params['gb'] = {'n_estimators': [100], 'random_state': [self.random_state]}
        self.params['bag'] = {'n_estimators': [10], 'random_state': [self.random_state]}
        self.params['qda'] = {'priors': [None]}
        self.params['lda'] = {'solver': 'svd'}
        self.params['svc'] = {'C': [1.0], 'random_state': [self.random_state]}
        self.params['out'] = {'code_size': [1.5]}
        self.params['gnb'] = {'prior': [None]}
        
        return True

    def write_results(self):
        with open(self.result_file, 'w') as f:
            json.dump(self.results, f)

    def MLkNN(self):
        model = MLkNN()
        opt = GridSearch(model, param_grid = self.params['mlknn'], seed = self.random_state)
        print(self.X_train.shape)
        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)
        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()
        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['MLKNN'] = scores
        print("========================================================================")
        print(scores)

    def LGB_multilabel(self):
        model = LGBMClassifier(objective='multiclass', random_state=5)
        opt = GridSearch(model, param_grid = self.params['LGB'], seed = self.random_state)
        print(self.X_train.shape)
        # opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)

        # y_valid_pred = opt.predict(self.X_valid)
        # y_valid_prob = opt.predict_proba(self.X_valid)
        
        
        # scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        # y_test_pred = opt.predict(self.X_test)
        # y_test_prob = opt.predict_proba(self.X_test)
        # scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        # scores = dict()
        # scores['valid'] = scores_valid
        # scores['test'] = scores_test

        # # Store the validation and test results

        # self.results['LGB'] = scores
        # print("========================================================================")
        # print(scores)

    
    

def main():
    
    md.get_data_sets()
    md.load_params()
    print(111)
    md.MLkNN()
    # md.LGB_multilabel()
    
    md.write_results()
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Training models")
    
    parser.add_argument('--train-features', action='store', dest='train-features', required=True, \
                        help='train-features file (numpyFile)')
    parser.add_argument('--validation-features', action='store', dest = 'validation-features', required=True,\
                        help='validation-features file (numpyFile)')
    parser.add_argument('--test-features', action='store', dest = 'test-features', required=True,\
                        help='test-features file (numpyFile)')

    args = vars(parser.parse_args())
    
    train = args['train-features']
    valid = args['validation-features']
    test = args['test-features']
    
    md = Model_Development(train, valid, test)

    main()
