import sys
sys.path.append('/home/chujunyi/MFDF/3_my_code')
import os
os.chdir('/home/chujunyi/MFDF/1_original_data/E')
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
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

np.set_printoptions(threshold = np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

with open('/home/chujunyi/MFDF/1_original_data/E/E_combine_origin_new_interactions.txt','r') as f:
    E_ddr_new_pair = f.readlines()
E_ddr_new_pair = [item.replace('\n','').split('\t') for item in E_ddr_new_pair]
E_ddr_new_pair = [item[0] + ',' + item[1] for item in E_ddr_new_pair]
#-------------------------

R_all_train_test = "e_admat_dgc_mat_2_line.txt"
(D,T,DT_signature,aAllPossiblePairs,dDs,dTs,diDs,diTs) = get_All_D_T_thier_Labels_Signatures(R_all_train_test)

DT_feature_pair_list = []
for index in diDs.values():
    for column in diTs.values():
        DT_feature_pair_list.append([index,column]) 
#-------------------------

def calc_metrics(y_results,DT_feature_pair_list,E_ddr_new_pair):
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
        mean_fp = np.mean(all_fp) * 10
        mean_fn = np.mean(all_fn) * 10           
        mean_tp = np.mean(all_tp) * 10
        mean_tn = np.mean(all_tn) * 10

        micro_precision = mean_tp / (mean_tp + mean_fp)
        micro_recall = mean_tp/ (mean_tp + mean_fn)
        micro_fscore = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        
        print('mean_fp = {}, mean_fn = {}, mean_tp = {}, mean_tn = {}'.format(mean_fp, mean_fn, mean_tp, mean_tn))
        print('micro_precision = {}, micro_recall = {}, micro_fscore = {}'.format(micro_precision, micro_recall, micro_fscore))
        
    y_choose_all.drop_duplicates(subset='test_idx_fold', keep='first', inplace=True) # (16, 5)
    y_choose_all['test_idx_fold'] = y_choose_all['test_idx_fold'].astype('int')

    y_choose_idx = [item for item in y_choose_all['test_idx_fold']]
    DTI_CDF_new_pair = [DT_feature_pair_list[i] for i in y_choose_idx]
    y_choose_all['new_pair'] = DTI_CDF_new_pair

    DTI_CDF_new_pair = [item[0] + ',' + item[1] for item in DTI_CDF_new_pair]
    
    ddr_intersection_dti = list(set(DTI_CDF_new_pair).intersection(set(E_ddr_new_pair)))
    print('DTI_CDF_new_pair: ', DTI_CDF_new_pair)
    print('ddr_intersection_dti = ', ddr_intersection_dti)
    
    return DTI_CDF_new_pair, y_choose_all, ddr_intersection_dti
	
def get_toy_config(rf_tree = 1, rf_max_depth = 1, rf_tree_2 = 1, rf_max_depth_2 = 1,
                   xgb_tree = 1, xgb_max_depth = 1, min_child_weight = 1, lr = 1,
                   xgb_tree_2 = 1, xgb_max_depth_2 = 1, min_child_weight_2 = 1, lr_2 = 1,
                   layer = 'rf1'):
    
    if layer == 'rf1' or layer == 'xgb1': 
        max_layers = 1
        n_folds = 1
    else: 
        max_layers, n_folds = 5, 10

    config = {}
    ca_config = {}
    ca_config["random_state"] = 1231
    ca_config["look_indexs_cycle"] = None
    ca_config["max_layers"] = max_layers
    ca_config["early_stopping_rounds"] = 1
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
        
    rf = {"n_folds": n_folds, "type": "RandomForestClassifier",
          "n_estimators": rf_tree, "max_depth": rf_max_depth, 'criterion' : 'entropy', 'bootstrap' : True,
          "n_jobs": -1, "criterion" : 'entropy', "class_weight":"balanced"}
    
    rf_2 = {"n_folds": n_folds, "type": "RandomForestClassifier",
          "n_estimators": rf_tree_2, "max_depth": rf_max_depth_2, 'criterion' : 'entropy', 'bootstrap' : True,
          "n_jobs": -1, "criterion" : 'entropy', "class_weight":"balanced"}
    
    xgb = {"n_folds": n_folds, "type": "XGBClassifier",'base_score':0.5, 'booster':'gbtree', 'colsample_bylevel':1,
           'colsample_bytree':1, 'eval_metric':'auc', 'gamma':0, 'learning_rate':lr,
           'max_delta_step':0, 'max_depth':xgb_max_depth, 'min_child_weight':min_child_weight, 'missing':None,
           'n_estimators':xgb_tree, 'n_jobs':-1, 'nthread':-1,
           'objective':'binary:logistic', 'random_state':1231, 'reg_alpha':0,
           'reg_lambda':1, 'scale_pos_weight':1, 'seed':1231, 'silent':True,
           'subsample':1}

    xgb_2 = {"n_folds": n_folds, "type": "XGBClassifier",'base_score':0.5, 'booster':'gbtree', 'colsample_bylevel':1,
           'colsample_bytree':1, 'eval_metric':'auc', 'gamma':0, 'learning_rate':lr_2,
           'max_delta_step':0, 'max_depth':xgb_max_depth_2, 'min_child_weight':min_child_weight_2, 'missing':None,
           'n_estimators':xgb_tree_2, 'n_jobs':-1, 'nthread':-1,
           'objective':'binary:logistic', 'random_state':1231, 'reg_alpha':0,
           'reg_lambda':1, 'scale_pos_weight':1, 'seed':1231, 'silent':True,
           'subsample':1}     
        
    if layer == 'rf1':
        ca_config["estimators"].append(rf)
        
    elif layer == 'xgb1':
        ca_config["estimators"].append(xgb)
        
    elif layer == 'rf1xgb1':
        ca_config["estimators"].append(rf)
        ca_config["estimators"].append(xgb)
        
    elif layer == 'rf1xgb1_2':
        ca_config["estimators"].append(rf_2)
        ca_config["estimators"].append(xgb_2)
    
    elif layer == 'xgb2':
        ca_config["estimators"].append(xgb)
        ca_config["estimators"].append(xgb_2)
    
    elif layer == 'rf2':
        ca_config["estimators"].append(rf)
        ca_config["estimators"].append(rf_2) 
    
    elif layer == 'rf2xgb1':
        ca_config["estimators"].append(rf)
        ca_config["estimators"].append(rf_2) 
        ca_config["estimators"].append(xgb_2)
    
    elif layer == 'rf1xgb2':
        ca_config["estimators"].append(xgb)
        ca_config["estimators"].append(xgb_2)
        ca_config["estimators"].append(rf_2)
    
    elif layer == 'rf2xgb2':
        ca_config["estimators"].append(rf)
        ca_config["estimators"].append(rf_2) 
        ca_config["estimators"].append(xgb)
        ca_config["estimators"].append(xgb_2)     
        
    config["cascade"] = ca_config
    return config
	
def run_classification(mode, seed, X_train_10_fold, X_test_10_fold, y_train_10_fold, y_test_10_fold,test_idx_10_fold, train_idx_10_fold, layer):   
        
    if mode == 'p':
        xgb_param = {}
        xgb_param[1231] = {'xgb_max_depth':3, 'min_child_weight':0.001, 'lr':0.1, 'xgb_tree':800,
                           'xgb_max_depth_2':9, 'min_child_weight_2':0.1, 'lr_2':0.1, 'xgb_tree_2':400}

        xgb_param[8367] = {'xgb_max_depth':7, 'min_child_weight':0.001, 'lr':0.1, 'xgb_tree':100,
                           'xgb_max_depth_2':9, 'min_child_weight_2':0.1, 'lr_2':0.1, 'xgb_tree_2':800}

        xgb_param[22]   = {'xgb_max_depth':20, 'min_child_weight':0.1, 'lr':0.1, 'xgb_tree':200,
                           'xgb_max_depth_2':7, 'min_child_weight_2':0.01, 'lr_2':0.1, 'xgb_tree_2':800}

        xgb_param[1812] = {'xgb_max_depth':20, 'min_child_weight':0.1, 'lr':0.1, 'xgb_tree':200,
                           'xgb_max_depth_2':9, 'min_child_weight_2':0.001, 'lr_2':0.1, 'xgb_tree_2':800}

        xgb_param[4659] = {'xgb_max_depth':9, 'min_child_weight':0.1, 'lr':0.01, 'xgb_tree':800,
                           'xgb_max_depth_2':20, 'min_child_weight_2':0.001, 'lr_2':0.1, 'xgb_tree_2':400}
        
        rf_param = {}
        rf_param[1231] = {'rf_tree':100, 'rf_max_depth':3, 'rf_tree_2': 800, 'rf_max_depth_2': 20}
        rf_param[8367] = {'rf_tree':100, 'rf_max_depth':3, 'rf_tree_2': 100, 'rf_max_depth_2': 30}                   
        rf_param[22]   = {'rf_tree':100, 'rf_max_depth':3, 'rf_tree_2': 400, 'rf_max_depth_2': None}                   
        rf_param[1812] = {'rf_tree':100, 'rf_max_depth':3, 'rf_tree_2': 600, 'rf_max_depth_2': 20}                   
        rf_param[4659] = {'rf_tree':100, 'rf_max_depth':3, 'rf_tree_2': 600, 'rf_max_depth_2': None}
                         
        
    elif mode == 'D':    
        xgb_param = {}
        xgb_param[1231] = {'xgb_max_depth':9, 'min_child_weight':0.001, 'lr':0.1, 'xgb_tree':200,
                           'xgb_max_depth_2':7, 'min_child_weight_2':0.1, 'lr_2':0.1, 'xgb_tree_2':400}

        xgb_param[8367] = {'xgb_max_depth':5, 'min_child_weight':0.01, 'lr':0.1, 'xgb_tree':400,
                           'xgb_max_depth_2':9, 'min_child_weight_2':0.1, 'lr_2':0.1, 'xgb_tree_2':400}

        xgb_param[22]   = {'xgb_max_depth':9, 'min_child_weight':0.001, 'lr':0.1, 'xgb_tree':400,
                           'xgb_max_depth_2':20, 'min_child_weight_2':0.01, 'lr_2':0.1, 'xgb_tree_2':600}

        xgb_param[1812] = {'xgb_max_depth':5, 'min_child_weight':0.001, 'lr':0.1, 'xgb_tree':400,
                           'xgb_max_depth_2':20, 'min_child_weight_2':1, 'lr_2':0.1, 'xgb_tree_2':600}

        xgb_param[4659] = {'xgb_max_depth':5, 'min_child_weight':0.01, 'lr':0.1, 'xgb_tree':800,
                           'xgb_max_depth_2':9, 'min_child_weight_2':0.001, 'lr_2':0.1, 'xgb_tree_2':600}
        
        rf_param = {}
        rf_param[1231] = {'rf_tree':600, 'rf_max_depth':3, 'rf_tree_2': 600, 'rf_max_depth_2': 20}
        rf_param[8367] = {'rf_tree':300, 'rf_max_depth':3, 'rf_tree_2': 300, 'rf_max_depth_2': 20}                   
        rf_param[22]   = {'rf_tree':200, 'rf_max_depth':3, 'rf_tree_2': 800, 'rf_max_depth_2': 20}                   
        rf_param[1812] = {'rf_tree':300, 'rf_max_depth':3, 'rf_tree_2': 600, 'rf_max_depth_2': 20}                   
        rf_param[4659] = {'rf_tree':200, 'rf_max_depth':3, 'rf_tree_2': 600, 'rf_max_depth_2': 20}

    elif mode == 'T':    
        xgb_param = {}
        xgb_param[1231] = {'xgb_max_depth':7, 'min_child_weight':0.01, 'lr':0.1, 'xgb_tree':800,
                           'xgb_max_depth_2':20, 'min_child_weight_2':0.001, 'lr_2':0.1, 'xgb_tree_2':400}

        xgb_param[8367] = {'xgb_max_depth':7, 'min_child_weight':0.001, 'lr':0.1, 'xgb_tree':600,
                           'xgb_max_depth_2':20, 'min_child_weight_2':0.01, 'lr_2':0.1, 'xgb_tree_2':800}

        xgb_param[22]   = {'xgb_max_depth':9, 'min_child_weight':0.1, 'lr':0.1, 'xgb_tree':800,
                           'xgb_max_depth_2':20, 'min_child_weight_2':0.01, 'lr_2':0.1, 'xgb_tree_2':400}

        xgb_param[1812] = {'xgb_max_depth':9, 'min_child_weight':0.001, 'lr':0.1, 'xgb_tree':200,
                           'xgb_max_depth_2':20, 'min_child_weight_2':0.1, 'lr_2':0.1, 'xgb_tree_2':200}

        xgb_param[4659] = {'xgb_max_depth':7, 'min_child_weight':0.1, 'lr':0.1, 'xgb_tree':600,
                           'xgb_max_depth_2':9, 'min_child_weight_2':0.01, 'lr_2':0.1, 'xgb_tree_2':800}
        
        rf_param = {}
        rf_param[1231] = {'rf_tree':600, 'rf_max_depth':3, 'rf_tree_2': 300, 'rf_max_depth_2': 20}
        rf_param[8367] = {'rf_tree':600, 'rf_max_depth':3, 'rf_tree_2': 300, 'rf_max_depth_2': 30}                   
        rf_param[22]   = {'rf_tree':100, 'rf_max_depth':3, 'rf_tree_2': 600, 'rf_max_depth_2': 30}                   
        rf_param[1812] = {'rf_tree':600, 'rf_max_depth':3, 'rf_tree_2': 200, 'rf_max_depth_2': None}                   
        rf_param[4659] = {'rf_tree':400, 'rf_max_depth':3, 'rf_tree_2': 800, 'rf_max_depth_2': 30}

    xgb_tree, xgb_max_depth = xgb_param[seed]['xgb_tree'], xgb_param[seed]['xgb_max_depth']
    min_child_weight, lr =  xgb_param[seed]['min_child_weight'], xgb_param[seed]['lr']
    xgb_tree_2, xgb_max_depth_2 = xgb_param[seed]['xgb_tree_2'], xgb_param[seed]['xgb_max_depth_2']
    min_child_weight_2, lr_2 =  xgb_param[seed]['min_child_weight_2'], xgb_param[seed]['lr_2']
    
    rf_tree, rf_max_depth, rf_tree_2, rf_max_depth_2 = rf_param[seed]['rf_tree'], rf_param[seed]['rf_max_depth'], rf_param[seed]['rf_tree_2'], rf_param[seed]['rf_max_depth_2']
    
    result,test_true_predict_compare = run_classification_configuration(X_train_10_fold, X_test_10_fold, 
                                                                        y_train_10_fold, y_test_10_fold,
                                                                        test_idx_10_fold, train_idx_10_fold,
                                                                        rf_tree, rf_max_depth, 
                                                                        rf_tree_2, rf_max_depth_2,
                                                                        xgb_tree, xgb_max_depth, min_child_weight, lr, 
                                                                        xgb_tree_2, xgb_max_depth_2, min_child_weight_2, lr_2, 
                                                                        layer, mode, seed)
        
    print('************results : one_mode + one_seed + one_layer + one_parameter************')  
    print('Avg_AUPR_training = {:.4},Avg_AUPR_testing = {:.4},Avg_AUC_training = {:.4},Avg_AUC_testing = {:.4}'.format(
            result[0],result[1],result[3],result[4]))
    print('folds_AUPR_testing:', result[2])
    print('folds_AUPR_training:', result[6])
    print('folds_AUC_testing:', result[5])
    print('folds_AUC_training:',result[7])
    print('precision_testing = {}, recall_testing = {}, fscore_testing = {}'.format(result[8][0], result[8][1], result[8][2]))
    print('precision_training = {}, recall_training = {}, fscore_training = {}'.format(result[9][0], result[9][1], result[9][2]))
    #print('G_mean = ', result[0][12])
    print('************************************')  
    return result, test_true_predict_compare

def gain_results(seeds, mode_list, layer, only_PathCS_feature = False):

    aupr_list, auc_list = [], []
    precision_list, recall_list, fscore_list = [], [], []
    test_true_predict_compare_10cv_seeds_modes = []
    recall_25_list, recall_50_list, recall_100_list, recall_200_list, recall_400_list = [], [], [], [], []
#     recall_25_list, recall_50_list = [], []
    G_mean_list = []
    for mode in mode_list:
        
        trails_AUPRs, trails_AUCs = [], []
        trails_precisions, trails_recalls, trails_fscores = [], [], []
        trails_recall_25, trails_recall_50, trails_recall_100, trails_recall_200, trails_recall_400 = [], [], [], [], []
#         trails_recall_25, trails_recall_50 = [], []
        trials_G_means = []
        seeds_results = []
        test_true_predict_compare_10cv_seeds = []
        for seed in seeds:
            print ("---------GENERATE_FOLD_{}_{}-----------------------------------------------".format(mode, seed))

            filename = '/home/chujunyi/MFDF/3_my_code/E/data/E_folddata_X-Y_S' + str(mode) + '_seed' + str(seed) + '.npz'
            folddata_XY = np.load(filename)

            X_train_10_fold, X_test_10_fold  = folddata_XY['X_train_10_fold'],folddata_XY['X_test_10_fold']
            y_train_10_fold, y_test_10_fold = folddata_XY['y_train_10_fold'],folddata_XY['y_test_10_fold']
            train_idx_10_fold, test_idx_10_fold = folddata_XY['train_idx_10_fold'],folddata_XY['test_idx_10_fold']
            
            if only_PathCS_feature == True:
                if mode == 'p':
                    X_train_10_fold = map(lambda x : x[:,:12], X_train_10_fold)
                    X_test_10_fold = map(lambda x : x[:,:12], X_test_10_fold)
                else:
                    X_train_10_fold = map(lambda x : x[:,:10], X_train_10_fold)
                    X_test_10_fold = map(lambda x : x[:,:10], X_test_10_fold)
                
            print ('-------------------------------------------------THIS SEED FINISHED----------------------------------')

            results,test_true_predict_compare_10cv = run_classification(mode, seed, X_train_10_fold, X_test_10_fold, y_train_10_fold, 
                                                                        y_test_10_fold,test_idx_10_fold, train_idx_10_fold, layer)
            
            seeds_results.append(results)
            trails_AUPRs.extend(results[2]) # 5z
            trails_AUCs.extend(results[5]) # 50
            trails_precisions.append(results[8][0]) # 5
            trails_recalls.append(results[8][1]) # 5
            trails_fscores.append(results[8][2]) # 5
            test_true_predict_compare_10cv_seeds.append(test_true_predict_compare_10cv)
            trails_recall_25.append(results[10])
            trails_recall_50.append(results[11])
            trails_recall_100.append(results[12])
            trails_recall_200.append(results[13])
            trails_recall_400.append(results[14])
            trials_G_means.extend(results[12])###15
            
            aupr, roc_auc = np.mean(trails_AUPRs), np.mean(trails_AUCs)
            precision, recall, fscore = np.mean(trails_precisions), np.mean(trails_recalls), np.mean(trails_fscores)
#             recall_25, recall_50 = np.mean(trails_recall_25), np.mean(trails_recall_50)
            recall_25, recall_50, recall_100, recall_200, recall_400 = np.mean(trails_recall_25), np.mean(trails_recall_50), np.mean(trails_recall_100), np.mean(trails_recall_200), np.mean(trails_recall_400)
            G_mean = np.mean(trials_G_means)
            
            print( "################Results###################" )
            print('model_architecture:',layer)
            print( "Mode: %s" % mode )
            print( "Average: AUPR: %s" % aupr ) 
            print( "Average: AUC: %s" % roc_auc )
            print( "Average: precision = {}, recall = {}, fscore = {} ".format(precision, recall, fscore))
            print( "Average: G_mean = ", G_mean)
            
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
            print('G_mean = ', result_[12])
            print('')
            print( "###########################################")
            
        
        aupr_list.append(aupr) 
        auc_list.append(roc_auc)
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)
        recall_25_list.append(recall_25)
        recall_50_list.append(recall_50)
        recall_100_list.append(recall_100)
        recall_200_list.append(recall_200)
        recall_400_list.append(recall_400)
        G_mean_list.append(G_mean)
        test_true_predict_compare_10cv_seeds_modes.append(test_true_predict_compare_10cv_seeds)
        print( "################Results###################" )
        print('model_architecture:',layer)
        print( "Mode: %s" % mode_list )
        print( "Average AUPR: %s" % aupr_list ) 
        print( "Average AUC: %s" % auc_list )
        print( "Average precision = {} ".format(precision_list))
        print( "Average recall = {} ".format(recall_list))
        print( "Average f1score = {} ".format(fscore_list))
        print( "Average recall_25 = {} ".format(recall_25_list))
        print( "Average recall_50 = {} ".format(recall_50_list))
        print( "Average recall_100 = {} ".format(recall_100_list))
        print( "Average recall_200 = {} ".format(recall_200_list))
        print( "Average recall_400 = {} ".format(recall_400_list))
        print( "Average G_mean = ,", G_mean_list)
        print( "###########################################")
    
    return test_true_predict_compare_10cv_seeds_modes
	
def run_classification_configuration(X_train_10_fold, X_test_10_fold, y_train_10_fold, y_test_10_fold,test_idx_10_fold, train_idx_10_fold,
                                     rf_tree, rf_max_depth, rf_tree_2, rf_max_depth_2,
                                     xgb_tree, xgb_max_depth, min_child_weight, lr, xgb_tree_2, xgb_max_depth_2, min_child_weight_2, lr_2,
                                     layer, mode, seed):
    
    folds_AUC_testing, folds_AUPR_testing = [], []
    folds_AUC_training, folds_AUPR_training= [], []
    folds_metrics3_training, folds_metrics3_testing = [], []
    test_true_predict_compare, train_true_predict_compare = [], []
    folds_recall_25, folds_recall_50, folds_recall_100, folds_recall_200, folds_recall_400 = [], [], [], [], []
    folds_G_mean = []
    i = 0
    for X_train, X_test, y_train, y_test, test_idx_fold, train_idx_fold in zip(X_train_10_fold, X_test_10_fold, y_train_10_fold, y_test_10_fold, test_idx_10_fold, train_idx_10_fold):
        
        config = get_toy_config(rf_tree, rf_max_depth, rf_tree_2, rf_max_depth_2,
                                xgb_tree, xgb_max_depth, min_child_weight, lr, xgb_tree_2, xgb_max_depth_2, min_child_weight_2, lr_2,
                                layer)
        gc = GCForest(config)
        print(config)
        X_train_enc = gc.fit_transform(X_train, y_train)

        y_pred_train = gc.predict(X_train)
        y_predprob_train = gc.predict_proba(X_train)
        y_pred_test = gc.predict(X_test)
        y_predprob_test = gc.predict_proba(X_test)
        
        temp = pd.DataFrame([y_test, y_predprob_test[:,1],y_pred_test]).T.sort_values(by = 1, ascending = False)
        recall_25 = precision_recall_fscore_support(temp.iloc[:25,:][0], temp.iloc[:25,:][2], pos_label = 1, average = 'binary')[1]
        recall_50 = precision_recall_fscore_support(temp.iloc[:50,:][0], temp.iloc[:50,:][2], pos_label = 1, average = 'binary')[1]
        recall_100 = precision_recall_fscore_support(temp.iloc[:100,:][0], temp.iloc[:100,:][2], pos_label = 1, average = 'binary')[1]
        recall_200 = precision_recall_fscore_support(temp.iloc[:200,:][0], temp.iloc[:200,:][2], pos_label = 1, average = 'binary')[1]
        recall_400 = precision_recall_fscore_support(temp.iloc[:400,:][0], temp.iloc[:400,:][2], pos_label = 1, average = 'binary')[1]
        
        test_true_predict_compare.append([test_idx_fold, y_pred_test, y_test, y_predprob_test[:,0], y_predprob_test[:,1]]) #10-cv
        train_true_predict_compare.append([train_idx_fold, y_pred_train, y_train, y_predprob_train[:,0], y_predprob_train[:,1]]) #10-cv

#         with open(r'E:\11_MFDF_code\3_my_code\9_major_revised\NR\code\12\model\NR_PathCS_' + layer + '_S' + str(mode) + '_seed' + str(seed) + '_fold' + str(i) + '.pkl', "wb") as f1:
#             pickle.dump(gc, f1, pickle.HIGHEST_PROTOCOL)
                
        precision_training, recall_training, _ = precision_recall_curve(y_train, y_predprob_train[:,1], pos_label=1)
        precision_testing, recall_testing, _ =   precision_recall_curve(y_test, y_predprob_test[:,1], pos_label=1)    
        
        AUPR_training, AUPR_testing = auc(recall_training,precision_training), auc(recall_testing, precision_testing)
        AUC_training, AUC_testing = roc_auc_score(y_train, y_predprob_train[:,1]), roc_auc_score(y_test, y_predprob_test[:,1])
        
        metrics3_testing = precision_recall_fscore_support(y_test, y_pred_test, pos_label = 1, average = 'binary')[:3]
        metrics3_training = precision_recall_fscore_support(y_train, y_pred_train, pos_label = 1, average = 'binary')[:3]
                
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test, labels = [0,1]).ravel()
        specificity = float(tn) / float(tn + fp)
        recall = metrics3_testing[1]
        G_mean = np.sqrt(recall * specificity)

        folds_AUC_testing.append(AUC_testing)
        folds_AUPR_testing.append(AUPR_testing)
        folds_metrics3_testing.append(metrics3_testing)
        folds_AUC_training.append(AUC_training)
        folds_AUPR_training.append(AUPR_training)
        folds_metrics3_training.append(metrics3_training)
        folds_G_mean.append(G_mean)
        folds_recall_25.append(recall_25)
        folds_recall_50.append(recall_50)
        folds_recall_100.append(recall_100)
        folds_recall_200.append(recall_200)
        folds_recall_400.append(recall_400)
        i += 1
    Avg_AUPR_training = np.mean(folds_AUPR_training)
    Avg_AUPR_testing = np.mean(folds_AUPR_testing)
    Avg_AUC_training = np.mean(folds_AUC_training)
    Avg_AUC_testing = np.mean(folds_AUC_testing) 
    Avg_metrics3_training = np.mean(folds_metrics3_training, axis = 0)
    Avg_metrics3_testing = np.mean(folds_metrics3_testing, axis = 0)
    Avg_G_mean = np.mean(folds_G_mean)
    
    return [Avg_AUPR_training, Avg_AUPR_testing, folds_AUPR_testing,  #012
            Avg_AUC_training, Avg_AUC_testing, folds_AUC_testing, #345
            folds_AUPR_training, folds_AUC_training, #67
            Avg_metrics3_testing, Avg_metrics3_training, #89
            folds_recall_25, folds_recall_50, folds_recall_100, folds_recall_200, folds_recall_400, folds_G_mean], [test_true_predict_compare, train_true_predict_compare] #


mode_list = ["p", "D", "T"]
seeds = [1231, 8367, 22, 1812, 4659]
model_architecture = ['rf1', 'xgb1', 'rf2', 'xgb2', 'rf2xgb2', 'rf1xgb2', 'rf2xgb1', 'rf1xgb1_2','rf1xgb1'] # DNN time


for layer in model_architecture:
    print('model_architecture:',layer)
    time_begin_cpu = time.clock()
    time_begin_wall = time.time()
    rf1_PathCS_test_true_predict_compare_10cv_seeds_modes = gain_results(seeds,mode_list,layer, only_PathCS_feature = True)
    DTI_CDF_new_pair, y_choose_all, ddr_intersection_dti = calc_metrics(rf1_PathCS_test_true_predict_compare_10cv_seeds_modes,DT_feature_pair_list,E_ddr_new_pair)
    time_end_cpu = time.clock()
    time_end_wall = time.time()
    print('CPU Time = {}, Wall Time  = {}'.format(time_end_cpu - time_begin_cpu, time_end_wall - time_begin_wall))
