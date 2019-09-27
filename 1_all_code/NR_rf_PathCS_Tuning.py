
# coding: utf-8

# # 训练所需function

# In[2]:

import sys
sys.path.append(r'E:\11_MFDF_code\3_my_code')
import os
os.chdir(r'E:\11_MFDF_code\1_original_data\NR')
from Graph_utils import *
from functions import *
from SNF import *
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import auc,roc_auc_score,roc_curve,precision_recall_curve
from sklearn.preprocessing import MaxAbsScaler
from  sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import gcforest
from gcforest.gcforest import GCForest
import xgboost
from functools import reduce
import glob
from scipy import interp
import csv
import time
import random
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

np.set_printoptions(threshold = np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[3]:

with open(r'E:\11_MFDF_code\1_original_data\NR\NR_combine_origin_new_interactions.txt','r') as f:
    nr_ddr_new_pair = f.readlines()
nr_ddr_new_pair = [item.replace('\n','').split('\t') for item in nr_ddr_new_pair]
nr_ddr_new_pair = [item[0] + ',' + item[1] for item in nr_ddr_new_pair]
#-------------------------

R_all_train_test = "nr_admat_dgc_mat_2_line.txt"
(D,T,DT_signature,aAllPossiblePairs,dDs,dTs,diDs,diTs) = get_All_D_T_thier_Labels_Signatures(R_all_train_test)

DT_feature_pair_list = []
for index in diDs.values():
    for column in diTs.values():
        DT_feature_pair_list.append([index,column]) 
#-------------------------

def calc_metrics(y_results,DT_feature_pair_list,nr_ddr_new_pair):
    all_fp, all_fn = [],[]
    all_tp, all_tn = [],[]
    y_choose_all = pd.DataFrame()
    for mode in range(len(y_results)):
        for seed in range(len(y_results[mode])):
            for fold in range(len(y_results[mode][seed][0])): #[0] = test, [1] = train            
                a = pd.DataFrame()
                a['test_idx_fold'] = y_results[mode][seed][0][fold][0]
                a['y_predprob_test_0'] = y_results[mode][seed][0][fold][3]
                a['y_predprob_test_1'] = y_results[mode][seed][0][fold][4]
                a['y_pred_test'] = y_results[mode][seed][0][fold][1]
                a['y_true_test'] = y_results[mode][seed][0][fold][2]
                tn, fp, fn, tp = confusion_matrix(a['y_true_test'], a['y_pred_test'],labels = [0,1]).ravel()
                all_fp.append(fp)
                all_fn.append(fn)
                all_tp.append(tp)
                all_tn.append(tn)
                y_choose = a[a['y_pred_test'] == 0][a['y_true_test'] == 1]
                y_choose_all = pd.concat([y_choose_all, y_choose])
    
    y_choose_all.drop_duplicates(subset='test_idx_fold', keep='first', inplace=True) # (16, 5)
    y_choose_all['test_idx_fold'] = y_choose_all['test_idx_fold'].astype('int')

    y_choose_idx = [item for item in y_choose_all['test_idx_fold']]
    DTI_CDF_new_pair = [DT_feature_pair_list[i] for i in y_choose_idx]
    y_choose_all['new_pair'] = DTI_CDF_new_pair

    DTI_CDF_new_pair = [item[0] + ',' + item[1] for item in DTI_CDF_new_pair]
    
    mean_fp = np.mean(all_fp) * 10
    mean_fn = np.mean(all_fn) * 10
    mean_tp = np.mean(all_tp) * 10
    mean_tn = np.mean(all_tn) * 10
    
    ddr_intersection_dti = list(set(DTI_CDF_new_pair).intersection(set(nr_ddr_new_pair)))
    print('mean_fp = {}, mean_fn = {}, mean_tp = {}, mean_tn = {}'.format(mean_fp, mean_fn, mean_tp, mean_tn))
    print('DTI_CDF_new_pair: ', DTI_CDF_new_pair)
    print('ddr_intersection_dti = ', ddr_intersection_dti)
    
    return mean_fp, mean_fn, mean_tp, mean_tn, DTI_CDF_new_pair, y_choose_all, ddr_intersection_dti

# In[4]:

def gain_results(seeds, mode_list, layer, only_PathCS_feature = False):
    aupr_list = [] 
    auc_list = []
    metrics3_list = []
    test_true_predict_compare_10cv_seeds_modes = []
    for mode in mode_list:
        trails_AUPRs = []
        trails_AUCs = []
        metrics3 = []
        seeds_results = []
        test_true_predict_compare_10cv_seeds = []
        for seed in seeds:
            print ("---------GENERATE_FOLD_{}-----------------------------------------------".format(seed))

            filename = r'E:\11_MFDF_code\3_my_code\9_major_revised\NR\data\RFE100_NR_folddata_X-Y_S' + str(mode) + '_seed' + str(seed) + '.npz'
            folddata_XY = np.load(filename)

            X_train_10_fold, X_test_10_fold  = folddata_XY['X_train_rfe_10_fold'],folddata_XY['X_test_rfe_10_fold']
            y_train_10_fold, y_test_10_fold = folddata_XY['y_train_10_fold'],folddata_XY['y_test_10_fold']
            test_idx_10_fold, train_idx_10_fold = folddata_XY['train_idx_10_fold'],folddata_XY['test_idx_10_fold']
            
            if only_PathCS_feature == True:
                if mode == 'p':
                    X_train_10_fold = map(lambda x : x[:,:12], X_train_10_fold)
                    X_test_10_fold = map(lambda x : x[:,:12], X_test_10_fold)
                else:
                    X_train_10_fold = map(lambda x : x[:,:10], X_train_10_fold)
                    X_test_10_fold = map(lambda x : x[:,:10], X_test_10_fold)
                
            print ('-------------------------------------------------THIS SEED FINISHED----------------------------------')

            results,parameter_result_df,test_true_predict_compare_10cv = run_classification(X_train_10_fold, X_test_10_fold, y_train_10_fold, y_test_10_fold,test_idx_10_fold, train_idx_10_fold,layer)
            seeds_results.append(results)
            trails_AUPRs.extend(results[2])
            trails_AUCs.extend(results[5])
            metrics3.extend(results[8])
            
            test_true_predict_compare_10cv_seeds.append(test_true_predict_compare_10cv)
            aupr,c1 = mean_confidence_interval(trails_AUPRs) 
            roc_auc,c1 = mean_confidence_interval(trails_AUCs)
            
            print( "################Results###################" )
            print('model_architecture:',layer)
            print( "Mode: %s" % mode )
            print( "Average: AUPR: %s" % aupr ) 
            print( "Average: AUC: %s" % roc_auc )
            print( "Average: precision = {}, recall = {}, fscore = {} ".format(metrics3[0], metrics3[1], metrics3[2]))
            
        for result_ in seeds_results:
            print('seed_results: ')
            print('Avg_AUPR_training:',result_[0])
            print('Avg_AUPR:',result_[1])
            print('folds_AUPR:',result_[2])
            print('Avg_AUC_training:',result_[3])
            print('Avg_AUC:',result_[4])
            print('folds_AUC:',result_[5])
            print('precision_testing = {}, recall_testing = {}, fscore_testing = {}'.format(result_[8][0], result_[8][1], result_[8][2]))
            print('precision_training = {}, recall_training = {}, fscore_training = {}'.format(result_[9][0],result_[9][1], result_[9][2]))
            print('')
            print( "###########################################")
            
        
        aupr_list.append(aupr) 
        auc_list.append(roc_auc)
        metrics3_list.append(metrics3)
        print(metrics3_list)
        precision_list = np.array(metrics3_list[0])[range(0,len(seeds)*3,3)]
        recall_list = np.array(metrics3_list[0])[range(1,len(seeds)*3,3)]
        fscore_list = np.array(metrics3_list[0])[range(2,len(seeds)*3,3)]
        test_true_predict_compare_10cv_seeds_modes.append(test_true_predict_compare_10cv_seeds)
        print( "################Results###################" )
        print('model_architecture:',layer)
        print( "Mode: %s" % mode_list )
        print( "Average AUPR: %s" % aupr_list ) 
        print( "Average AUC: %s" % auc_list )
        print( "precision = {}, recall = {}, fscore = {} ".format(precision_list,recall_list,fscore_list))
        print( "###########################################")
    
    return aupr_list,auc_list,test_true_predict_compare_10cv_seeds_modes


# In[9]:

# params_1 = max_depth, min_child_weight
def get_toy_config(trees, max_depth, layer = 'rf1', max_layers = 1):
    config = {}
    ca_config = {}
    ca_config["random_state"] = 1231
    ca_config["look_indexs_cycle"] = None
    ca_config["data_save_rounds"] = 2
    ca_config["data_save_dir"] = r'E:\11_MFDF_code\4_v2_generate_dataset\NR\save_dir'
    ca_config["max_layers"] = max_layers
    ca_config["early_stopping_rounds"] = 1
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    
    rf = {"n_folds": 1, "type": "RandomForestClassifier",
          "n_estimators": trees, "max_depth": max_depth, 
          "n_jobs": -1, "criterion" : 'entropy', "class_weight":"balanced"}

    if layer == 'rf1':
        ca_config["estimators"].append(rf)

    config["cascade"] = ca_config
    return config



# In[18]:

def run_classification(X_train_10_fold, X_test_10_fold, y_train_10_fold, y_test_10_fold,test_idx_10_fold, train_idx_10_fold,layer):   
    
    max_depths = [3] # 13
    no_trees = [100,200,300] # 16
#     min_samples_splits = range(50,201,20)

    result = []
    parameter_list = []
    test_true_predict_compare_10cv = []
    for max_depth in max_depths:
        for tree in no_trees:
            parameter_result,test_true_predict_compare = run_classification_configuration(X_train_10_fold, X_test_10_fold, y_train_10_fold, y_test_10_fold,test_idx_10_fold, train_idx_10_fold,tree,max_depth,layer)
            result.append(parameter_result)
            test_true_predict_compare_10cv.append(test_true_predict_compare)

            parameter_list.append([[layer,max_depth,tree],parameter_result[0],parameter_result[1]])
                
            
    parameter_result_df = pd.DataFrame(parameter_list)
    parameter_result_df.columns = ['parameter','train_AUPR','Avg_AUPR']   
    print(parameter_result_df.sort_values(by='Avg_AUPR', axis=0))
    
    best_parameter_idx = parameter_result_df['Avg_AUPR'].idxmax()
    best_parameter_result = parameter_result_df.iloc[best_parameter_idx,:]
    
    print('best_parameter_idx : ',best_parameter_idx)
    print('best_parameter_result : ',best_parameter_result)
        
    result.sort(key=lambda x:(x[0],x[1]),reverse=True)
    
    print(result[0])
    print('Avg_AUPR_training = {:.4},Avg_AUPR_testing = {:.4},Avg_AUC_training = {:.4},Avg_AUC_testing = {:.4}'.format(
            result[0][0],result[0][1],result[0][3],result[0][4]))
    print('folds_AUPR_testing:', result[0][2])
    print('folds_AUPR_training:', result[0][6])
    print('folds_AUC_testing:', result[0][5])
    print('folds_AUC_training:',result[0][7])
    print('precision_testing = {}, recall_testing = {}, fscore_testing = {}'.format(result[0][8][0], result[0][8][1], result[0][8][2]))
    print('precision_training = {}, recall_training = {}, fscore_training = {}'.format(result[0][9][0], result[0][9][1], result[0][9][2]))
    print('************results************')  
    return result[0], parameter_result_df, test_true_predict_compare_10cv[best_parameter_idx]


# In[19]:

def run_classification_configuration(X_train_10_fold, X_test_10_fold, y_train_10_fold, y_test_10_fold,test_idx_10_fold, train_idx_10_fold,tree,max_depth,layer):
    
    
    i = 0
    folds_AUC_testing = []
    folds_AUPR_testing = []
    folds_AUC_training = []
    folds_AUPR_training = []
    folds_metrics3_training, folds_metrics3_testing = [], []
    test_true_predict_compare = []
    train_true_predict_compare = []
    for X_train, X_test, y_train, y_test, test_idx_fold, train_idx_fold in zip(X_train_10_fold, X_test_10_fold, y_train_10_fold, y_test_10_fold, test_idx_10_fold, train_idx_10_fold):

        config = get_toy_config(tree,max_depth, layer)
        gc = GCForest(config)
#         print(config)
        X_train_enc = gc.fit_transform(X_train, y_train, X_test, y_test)

        y_pred_train = gc.predict(X_train)
        y_predprob_train = gc.predict_proba(X_train)
        y_pred_test = gc.predict(X_test)
        y_predprob_test = gc.predict_proba(X_test)
        
        test_true_predict_compare.append([test_idx_fold, y_pred_test, y_test, y_predprob_test[:,0], y_predprob_test[:,1]]) #10-cv
        train_true_predict_compare.append([train_idx_fold, y_pred_train, y_train, y_predprob_train[:,0], y_predprob_train[:,1]]) #10-cv
        
                
        precision_training, recall_training, _ = precision_recall_curve(y_train, y_predprob_train[:,1], pos_label=1)
        precision_testing, recall_testing, _ =   precision_recall_curve(y_test, y_predprob_test[:,1], pos_label=1)    
        AUPR_training = auc(recall_training,precision_training)
        AUPR_testing = auc(recall_testing, precision_testing)
        AUC_training = roc_auc_score(y_train, y_predprob_train[:,1]) 
        AUC_testing = roc_auc_score(y_test, y_predprob_test[:,1]) 
        metrics3_testing = precision_recall_fscore_support(y_pred_test, y_test, pos_label = 1, average = 'binary')[:3]
        metrics3_training = precision_recall_fscore_support(y_pred_train, y_train, pos_label = 1, average = 'binary')[:3]
        
        folds_AUC_testing.append(AUC_testing)
        folds_AUPR_testing.append(AUPR_testing)
        folds_metrics3_testing.append(metrics3_testing)
        folds_AUC_training.append(AUC_training)
        folds_AUPR_training.append(AUPR_training)
        folds_metrics3_training.append(metrics3_training)
        
    Avg_AUPR_training = np.mean(folds_AUPR_training)
    Avg_AUPR_testing = np.mean(folds_AUPR_testing)
    Avg_AUC_training = np.mean(folds_AUC_training)
    Avg_AUC_testing = np.mean(folds_AUC_testing) 
    Avg_metrics3_training = np.mean(folds_metrics3_training, axis = 0)
    Avg_metrics3_testing = np.mean(folds_metrics3_testing, axis = 0)
    
    return [Avg_AUPR_training,Avg_AUPR_testing,folds_AUPR_testing, 
            Avg_AUC_training,Avg_AUC_testing,folds_AUC_testing,
            folds_AUPR_training,folds_AUC_training,
            Avg_metrics3_testing,Avg_metrics3_training], [test_true_predict_compare,train_true_predict_compare]

# # 训练

# In[20]:

# 三类特征

# In[ ]:

mode_list = ["p"]
seeds = [1231, 8367, 22, 1812, 4659]
model_architecture = ['rf1']

for layer in model_architecture:
    print('model_architecture:',layer)
    aupr_list,auc_list,rf1_PathCS_test_true_predict_compare_10cv_seeds_modes = gain_results(seeds,mode_list,layer, only_PathCS_feature = True)
mean_fp, mean_fn, mean_tp, mean_tn, DTI_CDF_new_pair, y_choose_all, ddr_intersection_dti = calc_metrics(rf1_PathCS_test_true_predict_compare_10cv_seeds_modes,DT_feature_pair_list,nr_ddr_new_pair)

print('88888888888888888888888888888888888888888888888888888888')

mode_list = ["D"]
seeds = [1231, 8367, 22, 1812, 4659]
model_architecture = ['rf1']

for layer in model_architecture:
    print('model_architecture:',layer)
    aupr_list,auc_list,rf1_PathCS_test_true_predict_compare_10cv_seeds_modes = gain_results(seeds,mode_list,layer, only_PathCS_feature = True)
mean_fp, mean_fn, mean_tp, mean_tn, DTI_CDF_new_pair, y_choose_all, ddr_intersection_dti = calc_metrics(rf1_PathCS_test_true_predict_compare_10cv_seeds_modes,DT_feature_pair_list,nr_ddr_new_pair)

print('88888888888888888888888888888888888888888888888888888888')

mode_list = ["T"]
seeds = [1231, 8367, 22, 1812, 4659]
model_architecture = ['rf1']

for layer in model_architecture:
    print('model_architecture:',layer)
    aupr_list,auc_list,rf1_PathCS_test_true_predict_compare_10cv_seeds_modes = gain_results(seeds,mode_list,layer, only_PathCS_feature = True)
mean_fp, mean_fn, mean_tp, mean_tn, DTI_CDF_new_pair, y_choose_all, ddr_intersection_dti = calc_metrics(rf1_PathCS_test_true_predict_compare_10cv_seeds_modes,DT_feature_pair_list,nr_ddr_new_pair)



