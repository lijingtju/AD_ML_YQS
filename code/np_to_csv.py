# 导入模块
import numpy as np
import pandas as pd
def f(file_name):
    path_np = "/home/lijing/data/covid_19/mrmd_experiment1/H1N1/valid_test_features/"

    path_pd = "/home/lijing/data/covid_19/AD_YQS/code/data/"

    data_np = np.load(path_np + file_name)

    np_to_csv = pd.DataFrame(data_np)
    # print(np_to_csv.shape)

    np_to_csv.to_csv(path_pd + file_name[:-4]+".csv", index=False, index_label="index_label") 

fea_list = ['hashap', 'hashtt', 'maccs', 'avalon']
for fea in fea_list:
    file_train = fea + "-H1N1_stand44_edge_balance_train.npy"
    file_test = fea + "-H1N1_stand44_edge_balance_test.npy"
    f(file_train)
    f(file_test)