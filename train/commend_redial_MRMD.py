import os


def com(edge_list, des_list, cla_list):
    for edge in edge_list:
        for des, cla in zip(des_list, cla_list):
            try:
                if des == 'rdkDes':
                    print(
                    'python3 /home/lijing/data/covid_19/other_model/redial-2020/train/train_tune_added.py --train-features /home/lijing/data/covid_19/mrmd_experiment1/H1N1/rdk_descriptors/'+des+'-H1N1_stand'+str(edge)+'_edge_balance_train.npy --validation-features /home/lijing/data/covid_19/mrmd_experiment1/H1N1/rdk_descriptors/'+des+'-H1N1_stand'+str(edge)+'_edge_balance_val.npy --test-features /home/lijing/data/covid_19/mrmd_experiment1/H1N1/rdk_descriptors/'+des+'-H1N1_stand'+str(edge)+'_edge_balance_test.npy --classifier '+cla
                    )
                else:
                    print(
                    'python3 /home/lijing/data/covid_19/other_model/redial-2020/train/train_tune_added.py --train-features /home/lijing/data/covid_19/mrmd_experiment1/H1N1/valid_test_features/'+des+'-H1N1_stand'+str(edge)+'_edge_balance_train.npy --validation-features /home/lijing/data/covid_19/mrmd_experiment1/H1N1/valid_test_features/'+des+'-H1N1_stand'+str(edge)+'_edge_balance_val.npy --test-features /home/lijing/data/covid_19/mrmd_experiment1/H1N1/valid_test_features/'+des+'-H1N1_stand'+str(edge)+'_edge_balance_test.npy --classifier '+cla
                    )
            except Exception as e:
                pass
            continue

edge_list = [44]
des_list = ['lecfp6', 'lecfp6', 'lecfp6', 'tpatf', 'rdkDes']
cla_list = ['GB', 'SVC', 'ETAs', 'DT', 'RF']
com(edge_list, des_list, cla_list)


