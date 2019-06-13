
# coding: utf-8

# In[4]:

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
import xgboost
from functools import reduce
import glob
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import csv
np.set_printoptions(threshold=np.inf)


# In[2]:

labels = pd.read_csv(r'E:\11_MFDF_code\4_v2_generate_dataset\NR\NR_labels.csv')
labels = np.array(labels['0'])


# In[1]:

def run_classification_configuration(data,labels,test_idx,classifier):
    All_scores = []
    total_AUPR_training = 0
    total_AUPR_testing = 0
    total_AUC_training = 0
    total_AUC_testing = 0
    labels = np.array(labels)
    folds_AUPR = []
    folds_AUC = []
    test_true_predict_compare = []
    i = 0
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

        clf = classifier
        clf.fit(X_train_maxabs_transform, y_train)
        y_pred_train = clf.predict(X_train_maxabs_transform)
        y_predprob_train = clf.predict_proba(X_train_maxabs_transform)
        y_pred_test = clf.predict(X_test_maxabs_transform)
        y_predprob_test = clf.predict_proba(X_test_maxabs_transform)

        y_predprob_test_df = pd.DataFrame(y_predprob_test)
        y_predprob_train_df = pd.DataFrame(y_predprob_train)
        test_true_predict_compare.append([test_idx_fold, y_pred_test, y_test, y_predprob_test[:,0], y_predprob_test[:,1]]) #10-cv

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


        Avg_AUPR_training = 1.0*total_AUPR_training
        Avg_AUPR = 1.0*total_AUPR_testing/len(data)
        Avg_AUC_training = 1.0*total_AUC_training  #added   
        Avg_AUC = 1.0*total_AUC_testing/len(data)    #added   

    return [Avg_AUPR_training,Avg_AUPR,folds_AUPR,Avg_AUC_training,Avg_AUC,folds_AUC],test_true_predict_compare  #added AUC


# In[4]:

def gain_results(seeds,mode,labels, classifier):
    trails_AUPRs = []
    trails_AUCs = []
    seeds_results = []
    test_true_predict_compare_10cv_seeds = []
    for seed in seeds:
        print ("---------GENERATE_FOLD-----------------------------------------------")

        file_seq_folddata = glob.glob(r'E:\11_MFDF_code\4_v2_generate_dataset\NR\NR_folddata_S' + str(mode) + '_seed' + str(seed) + '_fold*' + '.csv')
        file_seq_testidx = glob.glob(r'E:\11_MFDF_code\4_v2_generate_dataset\NR\NR_testidx_S' + str(mode) + '_seed' + str(seed) + '_fold*' + '.csv')

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
        parameter_result,test_true_predict_compare_10cv = run_classification_configuration(data,labels,test_idx,classifier)
        seeds_results.append(parameter_result)
        trails_AUPRs.extend(parameter_result[2])
        trails_AUCs.extend(parameter_result[5])
        test_true_predict_compare_10cv_seeds.append(test_true_predict_compare_10cv)
        aupr,c1 = mean_confidence_interval(trails_AUPRs) 
        roc_auc = []
        auc_roc,c1 = mean_confidence_interval(trails_AUCs)
        roc_auc.append(auc_roc)
        
    print( "################Results###################" )
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


# In[40]:

def process_model(seeds,labels, classifiers, names):
    aupr_list = [] 
    auc_list = []
    test_true_predict_compare_10cv_seeds_classifiers = []
    for name,classifier in zip(names,classifiers):
        print('classifier: ',name)
        seeds_results,aupr,roc_auc,test_true_predict_compare_10cv_seeds = gain_results(seeds,mode,labels, classifier)
        aupr_list.append(aupr) 
        auc_list.append(roc_auc)
        test_true_predict_compare_10cv_seeds_classifiers.append(test_true_predict_compare_10cv_seeds)
    print( "################Results###################" )
    print( "Mode: %s" % mode_list )
    print( "Average AUPR: %s" % aupr_list ) 
    print( "Average AUC: %s" % auc_list )
    print( "###########################################")
    return aupr_list,auc_list,test_true_predict_compare_10cv_seeds_classifiers


# In[68]:

mode_list = ["p","D","T"]
seeds = [7771, 8367, 22, 1812, 4659]

names = ['RandomForestClassifier()', 'xgboost.XGBClassifier()', 'svm.SVC()', 'GradientBoostingClassifier()', 'LogisticRegression()',
              'MLPClassifier()', 'DecisionTreeClassifier()', 'AdaBoostClassifier()', 'GaussianNB()']

classifiers = [RandomForestClassifier(), xgboost.XGBClassifier(), svm.SVC(probability = True), GradientBoostingClassifier(), LogisticRegression(),
              MLPClassifier(), DecisionTreeClassifier(), AdaBoostClassifier(), GaussianNB()]

#names = ['RandomForestClassifier()', 'LogisticRegression()']

#classifiers = [RandomForestClassifier(),LogisticRegression()]

classifiers_aupr_list = []
classifiers_auc_list = []
classifiers_test_true_predict_compare_10cv_seeds_classifiers = []

for mode in mode_list:
    print('mode:',mode)
    aupr_list,auc_list,test_true_predict_compare_10cv_seeds_classifiers = process_model(seeds,labels, classifiers, names)
    classifiers_aupr_list.append(aupr_list)
    classifiers_auc_list.append(auc_list)
    classifiers_test_true_predict_compare_10cv_seeds_classifiers.append(test_true_predict_compare_10cv_seeds_classifiers)

with open(r'E:\11_MFDF_code\9_y_pred\training_NR_other_classifiers.csv', 'wb') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerow(['Each row: 1.name of classifiers; 2.classifiers; 3,classifiers_aupr; 4.classifiers_auc; 5.classifiers_y(mode,classifier,seed,cv,result,y_#5'])
    wr.writerow(names)
    wr.writerow(classifiers)
    wr.writerow(classifiers_aupr_list)
    wr.writerow(classifiers_auc_list)
    wr.writerow(classifiers_test_true_predict_compare_10cv_seeds_classifiers)


# In[43]:

classifiers_aupr_list


# In[46]:

print(len(classifiers_test_true_predict_compare_10cv_seeds_classifiers)) # 3 mode
print(len(classifiers_test_true_predict_compare_10cv_seeds_classifiers[0])) # 9 classifier
print(len(classifiers_test_true_predict_compare_10cv_seeds_classifiers[0][0])) # 5 seed
print(len(classifiers_test_true_predict_compare_10cv_seeds_classifiers[0][0][0])) # 10 cv
print(len(classifiers_test_true_predict_compare_10cv_seeds_classifiers[0][0][0][0])) # 5 results
print(len(classifiers_test_true_predict_compare_10cv_seeds_classifiers[0][0][0][0][0])) # 140


