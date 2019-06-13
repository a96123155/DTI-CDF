
# coding: utf-8

# In[1]:

import sys
sys.path.append('/home/dqw_cyy/11_MFDF_code/3_my_code')
import os
os.chdir('/home/dqw_cyy/11_MFDF_code/1_original_data/IC')
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
import matplotlib.pyplot as plt
from scipy import interp
import csv
np.set_printoptions(threshold=np.inf)


# In[2]:

labels = pd.read_csv('/home/dqw_cyy/11_MFDF_code/4_v2_generate_dataset/IC/IC_labels.csv')
labels = np.array(labels['0'])


# In[3]:

def get_toy_config(trees,cw,c,layer = 'rf2'):
    config = {}
    ca_config = {}
    ca_config["random_state"] = 1231
    ca_config["look_indexs_cycle"] = None
    ca_config["data_save_rounds"] = 0
    ca_config["data_save_dir"] = None
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    
    if layer == 'rf2':
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
    elif layer == 'xgb2':
        ca_config["estimators"].append({"n_folds": 1, "type": "XGBClassifier", "n_estimators": trees, "max_depth": 10,"objective": "binary:logistic", "silent": True, "nthread": -1, "learning_rate": cw} )
        ca_config["estimators"].append({"n_folds": 1, "type": "XGBClassifier", "n_estimators": trees, "max_depth": 10,"objective": "binary:logistic", "silent": True, "nthread": -1, "learning_rate": cw} )
    elif layer == 'rf3':
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
    elif layer == 'xgb3':
        ca_config["estimators"].append({"n_folds": 1, "type": "XGBClassifier", "n_estimators": trees, "max_depth": 10,"objective": "binary:logistic", "silent": True, "nthread": -1, "learning_rate": cw} )
        ca_config["estimators"].append({"n_folds": 1, "type": "XGBClassifier", "n_estimators": trees, "max_depth": 10,"objective": "binary:logistic", "silent": True, "nthread": -1, "learning_rate": cw} )
        ca_config["estimators"].append({"n_folds": 1, "type": "XGBClassifier", "n_estimators": trees, "max_depth": 10,"objective": "binary:logistic", "silent": True, "nthread": -1, "learning_rate": cw} )
    elif layer == 'rf1xgb1rf1':
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
        ca_config["estimators"].append({"n_folds": 1, "type": "XGBClassifier", "n_estimators": trees, "max_depth": 10,"objective": "binary:logistic", "silent": True, "nthread": -1, "learning_rate": cw} )
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
    elif layer == 'rf2xgb2rf2':
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
        ca_config["estimators"].append({"n_folds": 1, "type": "XGBClassifier", "n_estimators": trees, "max_depth": 10,"objective": "binary:logistic", "silent": True, "nthread": -1, "learning_rate": cw} )
        ca_config["estimators"].append({"n_folds": 1, "type": "XGBClassifier", "n_estimators": trees, "max_depth": 10,"objective": "binary:logistic", "silent": True, "nthread": -1, "learning_rate": cw} )
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
    else:
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
        ca_config["estimators"].append(
            {"n_folds": 1, "type": "XGBClassifier", "n_estimators": trees, "max_depth": 10,
             "objective": "binary:logistic", "silent": True, "nthread": -1, "learning_rate": cw} )
        ca_config["estimators"].append(
            {"n_folds": 1, "type": "XGBClassifier", "n_estimators": trees, "max_depth": 10,
             "objective": "binary:logistic", "silent": True, "nthread": -1, "learning_rate": cw} )
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
        ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": trees, "max_depth": None, "n_jobs": -1, "criterion" : c})
    config["cascade"] = ca_config
    return config


# In[4]:

def run_classification_configuration(data,labels,test_idx,trees,c,layer,performance=True,cw=0.001):
    All_scores = []
    total_AUPR_training = 0
    total_AUPR_testing = 0
    total_AUC_training = 0
    total_AUC_testing = 0
    labels = np.array(labels)
    folds_AUPR = []
    folds_AUC = []
    test_true_predict_compare = []
    for fold_data,test_idx_fold in zip(data,test_idx):
        length = len(fold_data)
        train_idx_fold = []
        for idx in range(length):
            if idx not in test_idx_fold['0']:
                train_idx_fold.append(idx)
        fold_data = np.array(fold_data) #(1404L, 488L)
        test_idx_fold = np.array(test_idx_fold['0']) #(140L,)
        train_idx_fold = np.array(train_idx_fold) #(1404L,)
        X_train, X_test = fold_data[train_idx_fold,], fold_data[test_idx_fold,]
        y_train, y_test = labels[train_idx_fold], labels[test_idx_fold]


        max_abs_scaler = MaxAbsScaler()
        X_train_maxabs_fit = max_abs_scaler.fit(X_train) 

        X_train_maxabs_transform = max_abs_scaler.transform(X_train)

        X_test_maxabs_transform = max_abs_scaler.transform(X_test)
        config = get_toy_config(trees,cw,c,layer)
        gc = GCForest(config)
        X_train_enc = gc.fit_transform(X_train_maxabs_transform, y_train)

        y_pred_train = gc.predict(X_train_maxabs_transform)
        y_predprob_train = gc.predict_proba(X_train_maxabs_transform)
        y_pred_test = gc.predict(X_test_maxabs_transform)
        y_predprob_test = gc.predict_proba(X_test_maxabs_transform)
        y_predprob_test_df = pd.DataFrame(y_predprob_test)
        y_predprob_train_df = pd.DataFrame(y_predprob_train)
        test_true_predict_compare.append([test_idx_fold, y_pred_test, y_test, y_predprob_test[:,0], y_predprob_test[:,1]]) #10-cv
        #test_true_predict_compare.append([test_idx_fold, y_pred_test, y_test, y_predprob_test]) #10-cv


        if performance:
            precision_training, recall_training, _ = precision_recall_curve(y_train, y_predprob_train[:,1:2], pos_label=1)
            precision_testing, recall_testing, _ =   precision_recall_curve(y_test, y_predprob_test[:,1:2], pos_label=1)
            AUPR_training = auc(recall_training,precision_training)
            AUPR_testing = auc(recall_testing, precision_testing)
            AUC_training = roc_auc_score(y_train, y_predprob_train[:,1:2]) #added
            AUC_testing = roc_auc_score(y_test, y_predprob_test[:,1:2]) #added
            folds_AUC.append(AUC_testing)
            folds_AUPR.append(AUPR_testing)

            #print AUPR_testing
            total_AUPR_training += AUPR_training
            total_AUPR_testing += AUPR_testing
            total_AUC_training += AUC_training #added
            total_AUC_testing += AUC_testing  #added
        else:
            All_scores.append(scores_testing)
    if performance:
        Avg_AUPR_training = 1.0*total_AUPR_training
        Avg_AUPR = 1.0*total_AUPR_testing/len(data)
        Avg_AUC_training = 1.0*total_AUC_training  #added   
        Avg_AUC = 1.0*total_AUC_testing/len(data)    #added   
        
    if performance:
        return [Avg_AUPR_training,Avg_AUPR,folds_AUPR,Avg_AUC_training,Avg_AUC,folds_AUC],test_true_predict_compare  #added AUC
    else:
        return All_scores, test_true_predict_compare


# In[5]:

def run_classification(data,labels,test_idx,learning_rate,max_depth,no_trees,criterion,layer):
    learning_rate = learning_rate
    max_depth = max_depth
    no_trees = no_trees
    c = "entropy"

    result = []
    parameter_list = []
    test_true_predict_compare_10cv = []
    if layer in ['xgb2','rf1xgb1rf1','xgb3','rf2xgb2rf2','rfxgb2rf']:
        for cw in learning_rate:
            for t in no_trees:
                parameter_result,test_true_predict_compare = run_classification_configuration(data,labels,test_idx,t,c,layer,cw)
                result.append(parameter_result)
                test_true_predict_compare_10cv.append(test_true_predict_compare)
                
                parameter_list.append([[layer,cw,t],parameter_result[1]])
                
    else:
        for t in no_trees:
            parameter_result,test_true_predict_compare = run_classification_configuration(data,labels,test_idx,t,c,layer)
            result.append(parameter_result)
            test_true_predict_compare_10cv.append(test_true_predict_compare)
            
            parameter_list.append([[t],parameter_result[1]])
            
    parameter_result_df = pd.DataFrame(parameter_list)
    parameter_result_df.columns = ['parameter','Avg_AUPR']   
    print(parameter_result_df)
    best_parameter_idx = parameter_result_df['Avg_AUPR'].idxmax()
    best_parameter_result = parameter_result_df.iloc[best_parameter_idx,:]
    
    print('best_parameter_idx : ',best_parameter_idx)
    print('best_parameter_result : ',best_parameter_result)
        
    result.sort(key=lambda x:x[1],reverse=True)
    print(result[0][1])
      
    return result[0], parameter_result_df, test_true_predict_compare_10cv[best_parameter_idx]


# In[6]:

def gain_results(seeds,mode,labels,learning_rate,max_depth,no_trees,criterion,layer):
    trails_AUPRs = []
    trails_AUCs = []
    seeds_results = []
    test_true_predict_compare_10cv_seeds = []
    for seed in seeds:
        print ("---------GENERATE_FOLD-----------------------------------------------")

        file_seq_folddata = glob.glob('/home/dqw_cyy/11_MFDF_code/4_v2_generate_dataset/IC/IC_folddata_S' + str(mode) + '_seed' + str(seed) + '_fold*' + '.csv')
        file_seq_testidx = glob.glob('/home/dqw_cyy/11_MFDF_code/4_v2_generate_dataset/IC/IC_testidx_S' + str(mode) + '_seed' + str(seed) + '_fold*' + '.csv')

        file_seq_folddata.sort()
        file_seq_testidx.sort()
        print(seed)

        data = []
        test_idx = []
        for i in range(10):
            fold_data = pd.read_csv(file_seq_folddata[i])
            testidx = pd.read_csv(file_seq_testidx[i])
            data.append(fold_data)
            test_idx.append(testidx)   

        print ('-------------------------------------------------THIS SEED FINISHED----------------------------------')
        results,parameter_result_df,test_true_predict_compare_10cv = run_classification(data,labels,test_idx,learning_rate,max_depth,no_trees,criterion,layer)
        seeds_results.append(results)
        trails_AUPRs.extend(results[2])
        trails_AUCs.extend(results[5])
        test_true_predict_compare_10cv_seeds.append(test_true_predict_compare_10cv)
        aupr,c1 = mean_confidence_interval(trails_AUPRs) 
        roc_auc = []
        auc_roc,c1 = mean_confidence_interval(trails_AUCs)
        roc_auc.append(auc_roc)
        
    print( "################Results###################" )
    print('model_architecture:',layer)
    print( "Mode: %s" % mode )
    print( "Average AUPR: %s" % aupr ) 
    print( "Average AUC: %s" % roc_auc )
    for result_ in seeds_results:
        print('seed_results: ')
        print('Avg_AUPR_training:',result_[0])
        print('Avg_AUPR:',result_[1])
        print('folds_AUPR:',result_[2])
        print('Avg_AUC_training:',result_[3])
        print('Avg_AUC:',result_[4])
        print('folds_AUC:',result_[5])
        print('')
    print( "###########################################")
    
    return seeds_results,aupr,roc_auc,test_true_predict_compare_10cv_seeds


# In[7]:

def process_model(mode_list,seeds,labels,learning_rate,max_depth,no_trees,criterion,layer):
    aupr_list = [] 
    auc_list = []
    test_true_predict_compare_10cv_seeds_modes = []
    for mode in mode_list:
        seeds_results,aupr,roc_auc,test_true_predict_compare_10cv_seeds = gain_results(seeds,mode,labels,learning_rate,max_depth,no_trees,criterion,layer)
        aupr_list.append(aupr) 
        auc_list.append(roc_auc)
        test_true_predict_compare_10cv_seeds_modes.append(test_true_predict_compare_10cv_seeds)
    print( "################Results###################" )
    print('model_architecture:',layer)
    print( "Mode: %s" % mode_list )
    print( "Average AUPR: %s" % aupr_list ) 
    print( "Average AUC: %s" % auc_list )
    print( "###########################################")
    return aupr_list,auc_list,test_true_predict_compare_10cv_seeds_modes


# In[ ]:

learning_rate = [0.001,0.01,0.1]
max_depth = [3,4,5,6]
no_trees = [50,150,250,500]
criterion = ["entropy"]

mode_list = ["p"]
seeds = [7771, 8367, 22, 1812, 4659]
model_architecture = ['rf1xgb1rf1']



for layer in model_architecture:
    print('model_architecture:',layer)
    aupr_list,auc_list,test_true_predict_compare_10cv_seeds_modes = process_model(mode_list,seeds,labels,learning_rate,max_depth,no_trees,criterion,layer)

with open('/home/dqw_cyy/11_MFDF_code/9_y_pred/6.1_IC_MFDF_rf1xgb1rf1.csv', 'wb') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerow(['Each_row: 1.mode; 2.layer; 3.aupr; 4.auc; 5.y'])
    wr.writerow(mode_list)
    wr.writerow(layer)
    wr.writerow(aupr_list)
    wr.writerow(auc_list)
    wr.writerow(test_true_predict_compare_10cv_seeds_modes)
        


# In[ ]:

learning_rate = [0.001,0.01,0.1]
max_depth = [3,4,5,6]
no_trees = [50,150,250,500]
criterion = ["entropy"]

mode_list = ["p","D","T"]
seeds = [7771, 8367, 22, 1812, 4659]
model_architecture = ['rf1xgb2rf1']

for layer in model_architecture:
    print('model_architecture:',layer)
    aupr_list,auc_list,test_true_predict_compare_10cv_seeds_modes = process_model(mode_list,seeds,labels,learning_rate,max_depth,no_trees,criterion,layer)

with open('/home/dqw_cyy/11_MFDF_code/9_y_pred/6.1_IC_MFDF_rf1xgb2rf1.csv', 'wb') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerow(['Each_row: 1.mode; 2.layer; 3.aupr; 4.auc; 5.y(mode,seed,cv,y_#5)'])
    wr.writerow(mode_list)
    wr.writerow(layer)
    wr.writerow(aupr_list)
    wr.writerow(auc_list)
    wr.writerow(test_true_predict_compare_10cv_seeds_modes)
        

# In[ ]:

learning_rate = [0.001,0.01,0.1]
max_depth = [3,4,5,6]
no_trees = [50,150,250,500]
criterion = ["entropy"]

mode_list = ["p","D","T"]
seeds = [7771, 8367, 22, 1812, 4659]
model_architecture = ['rf2xgb2rf2']

for layer in model_architecture:
    print('model_architecture:',layer)
    aupr_list,auc_list,test_true_predict_compare_10cv_seeds_modes = process_model(mode_list,seeds,labels,learning_rate,max_depth,no_trees,criterion,layer)

with open('/home/dqw_cyy/11_MFDF_code/9_y_pred/6.1_IC_MFDF_rf2xgb2rf2.csv', 'wb') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerow(['Each_row: 1.mode; 2.layer; 3.aupr; 4.auc; 5.y(mode,seed,cv,y_#5)'])
    wr.writerow(mode_list)
    wr.writerow(layer)
    wr.writerow(aupr_list)
    wr.writerow(auc_list)
    wr.writerow(test_true_predict_compare_10cv_seeds_modes)

# In[ ]:

print(len(test_true_predict_compare_10cv_seeds_modes)) #1 mode
print(len(test_true_predict_compare_10cv_seeds_modes[0])) #1 seed
print(len(test_true_predict_compare_10cv_seeds_modes[0][0])) # 10 cv
print(len(test_true_predict_compare_10cv_seeds_modes[0][0][0])) #5 results
print(len(test_true_predict_compare_10cv_seeds_modes[0][0][0][0])) #140
