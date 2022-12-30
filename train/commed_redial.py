import os


def com(edge_list, des_list, cla_list):
    for edge in edge_list:
        for des, cla in zip(des_list, cla_list):
            try:
                if des == 'rdkDes':
                    print(
                    'python3 /home/lijing/data/covid_19/other_model/redial-2020/train/train_tune_added.py --train-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/EV71/rdk_descriptors/'+des+'-EV71_stand'+str(edge)+'_edge_balance_train.npy --validation-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/EV71/rdk_descriptors/'+des+'-EV71_stand'+str(edge)+'_edge_balance_val.npy --test-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/EV71/rdk_descriptors/'+des+'-EV71_stand'+str(edge)+'_edge_balance_test.npy --classifier '+cla
                    )
                else:
                    print(
                    'python3 /home/lijing/data/covid_19/other_model/redial-2020/train/train_tune_added.py --train-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/EV71/valid_test_features/'+des+'-EV71_stand'+str(edge)+'_edge_balance_train.npy --validation-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/EV71/valid_test_features/'+des+'-EV71_stand'+str(edge)+'_edge_balance_val.npy --test-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/EV71/valid_test_features/'+des+'-EV71_stand'+str(edge)+'_edge_balance_test.npy --classifier '+cla
                    )
            except Exception as e:
                pass
            continue

edge_list = [16]
des_list = ['fcfp6', 'hashap', 'hashtt', 'tpatf', 'rdkDes']
cla_list = ['BAG', 'BAG', 'ADA', 'RF', 'DT']
com(edge_list, des_list, cla_list)




# def com_SARS_CPE(edge_list, des_list, cla_list):
#     for edge in edge_list:
#         for des, cla in zip(des_list, cla_list):
#             try:
#                 if des == 'rdkDes':
#                     print(
#                     'python3 /home/lijing/data/covid_19/other_model/redial-2020/train/train_tune_added.py --train-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/SARS_CPE/rdk_descriptors/'+des+'-SARS_CPE_stand'+str(edge)+'_edge_balance_train.npy --validation-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/SARS_CPE/rdk_descriptors/'+des+'-SARS_CPE_stand'+str(edge)+'_edge_balance_val.npy --test-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/SARS_CPE/rdk_descriptors/'+des+'-SARS_CPE_stand'+str(edge)+'_edge_balance_test.npy --classifier '+cla
#                     )
#                 else:
#                     print(
#                     'python3 /home/lijing/data/covid_19/other_model/redial-2020/train/train_tune_added.py --train-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/SARS_CPE/valid_test_features/'+des+'-SARS_CPE_stand'+str(edge)+'_edge_balance_train.npy --validation-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/SARS_CPE/valid_test_features/'+des+'-SARS_CPE_stand'+str(edge)+'_edge_balance_val.npy --test-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/SARS_CPE/valid_test_features/'+des+'-SARS_CPE_stand'+str(edge)+'_edge_balance_test.npy --classifier '+cla
#                     )
#             except Exception as e:
#                 pass
#             continue

# edge_list = [35]
# des_list = ['rdk6', 'ecfp4', 'ecfp4', 'tpatf', 'rdkDes']
# cla_list = ['BAG', 'SVC', 'KNB', 'ETAs', 'ETAs']
# com_SARS_CPE(edge_list, des_list, cla_list)



# def com_H1N1(edge_list, des_list, cla_list):
#     for edge in edge_list:
#         for des, cla in zip(des_list, cla_list):
#             try:
#                 if des == 'rdkDes':
#                     print(
#                     'python3 /home/lijing/data/covid_19/other_model/redial-2020/train/train_tune_added.py --train-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/H1N1/rdk_descriptors/'+des+'-H1N1_stand'+str(edge)+'_edge_balance_train.npy --validation-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/H1N1/rdk_descriptors/'+des+'-H1N1_stand'+str(edge)+'_edge_balance_val.npy --test-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/H1N1/rdk_descriptors/'+des+'-H1N1_stand'+str(edge)+'_edge_balance_test.npy --classifier '+cla
#                     )
#                 else:
#                     print(
#                     'python3 /home/lijing/data/covid_19/other_model/redial-2020/train/train_tune_added.py --train-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/H1N1/valid_test_features/'+des+'-H1N1_stand'+str(edge)+'_edge_balance_train.npy --validation-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/H1N1/valid_test_features/'+des+'-H1N1_stand'+str(edge)+'_edge_balance_val.npy --test-features /home/lijing/data/covid_19/COVID19_SFY/add_wet_train_model/H1N1/valid_test_features/'+des+'-H1N1_stand'+str(edge)+'_edge_balance_test.npy --classifier '+cla
#                     )
#             except Exception as e:
#                 pass
#             continue

# edge_list = [25]
# des_list = ['laval', 'laval', 'laval', 'tpatf', 'rdkDes']
# cla_list = ['SVC', 'RF', 'DT', 'BAG', 'RF']
# com_H1N1(edge_list, des_list, cla_list)

