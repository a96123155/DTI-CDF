
# coding: utf-8

# In[1]:

import sys
sys.path.append('/home/yanyichu/11_MFDF_code/3_my_code')
import os
os.chdir('/home/yanyichu/11_MFDF_code/1_original_data/NR')
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
from functools import reduce


# In[2]:

def get_similarities(sim_file,dMap):
    sim = []
    for line in open(sim_file).readlines():
        edge_list = get_edge_list(line.strip())
        sim.append(make_sim_matrix(edge_list,dMap))
    return sim


# In[3]:

def get_features_per_fold(R_train,D_sim,T_sim, pair):

    accum_DDD,max_DDD = get_two_hop_similarities(D_sim,D_sim)
    accum_TTT,max_TTT = get_two_hop_similarities(T_sim,T_sim)

    accum_DDT,max_DDT = get_drug_relation(D_sim,R_train) 
    accum_DTT,max_DTT = get_relation_target(T_sim,R_train)

    accum_DDDT,_ = get_drug_relation(accum_DDD,R_train)
    _,max_DDDT = get_drug_relation(max_DDD,R_train)

    accum_DTTT,_ = get_relation_target(accum_TTT,R_train)
    _,max_DTTT = get_relation_target(max_TTT,R_train)

    accum_DTDT,max_DTDT = get_DTDT(R_train)

    accum_DDTT,max_DDTT = get_DDTT(R_train,D_sim,T_sim)

    features = []

    features.append(mat2vec(accum_DDT))
    features.append(mat2vec(max_DDT))
    features.append(mat2vec(accum_DTT))
    features.append(mat2vec(max_DTT))
    features.append(mat2vec(accum_DDDT))
    features.append(mat2vec(max_DDDT))

    if pair:
        features.append(mat2vec(accum_DTDT))
        features.append(mat2vec(max_DTDT))

    features.append(mat2vec(accum_DTTT))
    features.append(mat2vec(max_DTTT))
    features.append(mat2vec(accum_DDTT))
    features.append(mat2vec(max_DDTT))

    return features


# In[4]:

def generate_10_cv_data(mode,seed,cv_data,DT,D_sim,T_sim,diDs,diTs):
    labels = mat2vec(DT)
    test_idx = []
    folds_features = []
    for fold in cv_data[seed]:
        print ("---------GENERATE_FOLD-----------------------------------------------")
        if mode == "T":
            R_train = mask_matrix(DT,fold[1],True)
        else:
            R_train = mask_matrix(DT,fold[1]) #by default transpose is false

        DT_impute_D = impute_zeros(R_train,D_sim[0])
        DT_impute_T = impute_zeros(np.transpose(R_train),T_sim[0])

        GIP_D = Get_GIP_profile(np.transpose(DT_impute_D),"d")
        GIP_T = Get_GIP_profile(DT_impute_T,"t")


        WD = []
        WT = []

        for s in D_sim:
            WD.append(s)
        WD.append(GIP_D)

        for s in T_sim:
            WT.append(s)
        WT.append(GIP_T)
        D_SNF = SNF(WD,3,2)
        T_SNF = SNF(WT,3,2)

        DS_D = FindDominantSet(D_SNF,5)
        DS_T = FindDominantSet(T_SNF,5)

        np.fill_diagonal(DS_D,0)
        np.fill_diagonal(DS_T,0)

        features = get_features_per_fold(R_train,DS_D,DS_T, pair)

        labels = mat2vec(DT) 
        
        features_zip = list(zip(*features))
        folds_features.append(features_zip)

        if mode == "T":
            test_idx.append([j*col+i for (i,j) in fold[1]])
        else:
            test_idx.append([i*col+j for (i,j) in fold[1]])

    print ('-------------------------------------------------THIS SEED FINISHED----------------------------------')
    return folds_features,labels,test_idx


# In[5]:

R_all_train_test = "NR_combine_origin_new_interactions.txt"
D_sim_file = "nr_D_similarities.txt"
T_sim_file = "nr_T_similarities.txt"


# In[6]:

#### main script#################

(D,T,DT_signature,aAllPossiblePairs,dDs,dTs,diDs,diTs) = get_All_D_T_thier_Labels_Signatures(R_all_train_test)

R = get_edge_list(R_all_train_test)
DT = get_adj_matrix_from_relation(R,dDs,dTs)

D_sim = get_similarities(D_sim_file,dDs)
T_sim = get_similarities(T_sim_file,dTs)

row,col = DT.shape

all_matrix_index = []

for i in range(row):
    for j in range(col):
        all_matrix_index.append([i,j])

#### main script_OVER#################


# In[7]:

mode_list = ["p","D","T"]
seeds = [7771, 8367, 22, 1812, 4659]


# In[8]:

labels = mat2vec(DT)
pd.DataFrame(labels).to_csv('/home/yanyichu/11_MFDF_code/4_generate_dataset_DDR/NR/NR_DDR_labels.csv', index = False)


# In[10]:

for mode in mode_list:
    if mode == "p":
        cv_data = cross_validation(DT,seeds,1,10)
        pair = True
    elif mode == "D":
        cv_data = cross_validation(DT,seeds,0,10)
        pair = False
    elif mode == "T":
        pair = False
        cv_data = cross_validation(np.transpose(DT),seeds,0,10)

    #labels = mat2vec(DT)

    for seed in seeds:
        folds_features,labels,test_idx = generate_10_cv_data(mode,seed,cv_data,DT,D_sim,T_sim,diDs,diTs)
        i = 0
        for fold_data,test_idx_fold in zip(folds_features,test_idx):
            pd.DataFrame(fold_data).to_csv('/home/yanyichu/11_MFDF_code/4_generate_dataset_DDR/NR/NR_DDR_folddata_S' + str(mode) + '_seed' + str(seed) + '_fold' + str(i) + '.csv',index = False)
            pd.DataFrame(test_idx_fold).to_csv('/home/yanyichu/11_MFDF_code/4_generate_dataset_DDR/NR/NR_DDR_testidx_S' + str(mode) + '_seed' + str(seed) + '_fold' + str(i) + '.csv', index = False)
            i += 1

