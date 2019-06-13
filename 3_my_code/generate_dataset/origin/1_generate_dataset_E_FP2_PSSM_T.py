
# coding: utf-8

# In[1]:

import sys
sys.path.append('/lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_my_code')
import os
os.chdir('/lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/1_original_data/E')
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

for i in range(1000):
    print 'couzicocuziCOUZICOUZI',
print()

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

def merge_properties(dataset,molecule_feature_list = [],protein_feature_list = []):
    if molecule_feature_list != [] and protein_feature_list == []:
        molecule_feature_list = [dataset] + molecule_feature_list
        dataset_ = reduce(lambda x,y : pd.merge(x,y,on='compound_ID'),molecule_feature_list)
        return dataset_
    elif molecule_feature_list == [] and protein_feature_list != []:
        protein_feature_list = [dataset] + protein_feature_list
        dataset_ = reduce(lambda x,y : pd.merge(x,y,on='hsa_protein_ID'),protein_feature_list)
        return dataset_
    elif molecule_feature_list != [] and protein_feature_list != []:
        molecule_feature_list = [dataset] + molecule_feature_list
        dataset_ = reduce(lambda x,y : pd.merge(x,y,on='compound_ID'),molecule_feature_list)
        protein_feature_list = [dataset_] + protein_feature_list
        dataset_ = reduce(lambda x,y : pd.merge(x,y,on='hsa_protein_ID'),protein_feature_list)
        return dataset_
    else:
        return dataset


# In[5]:

def generate_10_cv_data(mode,seed,cv_data,DT,D_sim,T_sim,diDs,diTs,molecule_feature_list,protein_feature_list):
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
        features_pd = pd.DataFrame(features)
        feature_pd = pd.DataFrame(np.transpose(features))

        labels = mat2vec(DT) 
        DT_feature_pair_list = []
        for index in diDs.values():
            for column in diTs.values():
                DT_feature_pair_list.append([index,column]) 
        DT_feature_pd = pd.concat([pd.DataFrame(DT_feature_pair_list),pd.DataFrame(labels)],axis=1)
        DT_feature_pd.columns = ['compound_ID','hsa_protein_ID','label']
        DT_feature_pd = pd.concat([DT_feature_pd,feature_pd],axis=1)

        for i in range(DT_feature_pd['hsa_protein_ID'].shape[0]):
            DT_feature_pd.loc[i,'hsa_protein_ID'] = DT_feature_pd.loc[i,'hsa_protein_ID'].lstrip('hsa')
        if mode == "p":
            DT_feature_pd.columns = ['compound_ID','hsa_protein_ID','label','ddr_0','ddr_1','ddr_2','ddr_3','ddr_4','ddr_5','ddr_6','ddr_7','ddr_8','ddr_9','ddr_10','ddr_11']
        else:
            DT_feature_pd.columns = ['compound_ID','hsa_protein_ID','label','ddr_0','ddr_1','ddr_2','ddr_3','ddr_4','ddr_5','ddr_6','ddr_7','ddr_8','ddr_9']

        DT_feature_pd_merged = merge_properties(DT_feature_pd,molecule_feature_list,protein_feature_list) 
        labels = DT_feature_pd_merged['label']

        DT_feature_pd_merged[col_name] = preprocessing.scale(DT_feature_pd_merged[col_name])
        features_merge_pd = np.transpose(DT_feature_pd_merged.drop(['compound_ID','hsa_protein_ID','label'],axis=1)).reset_index(drop=True)

        features_merge_np = np.array(features_merge_pd)
        features_zip = list(zip(*features_merge_np))
        folds_features.append(features_zip)

        if mode == "T":
            test_idx.append([j*col+i for (i,j) in fold[1]])
        else:
            test_idx.append([i*col+j for (i,j) in fold[1]])

    print ('-------------------------------------------------THIS SEED FINISHED----------------------------------')
    return folds_features,labels,test_idx


# In[6]:

protein_psepssm_file = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/2_feature/E/e_psepssm.csv')
molecule_fp2_file = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/2_feature/E/E_fp2.csv')

molecule_fp2_file['compound_ID'] = molecule_fp2_file['compound_ID'].astype('str')
protein_psepssm_file['hsa_protein_ID'] = protein_psepssm_file['hsa_protein_ID'].astype('str')

col_name_fp2 = list(molecule_fp2_file.columns)
col_name_psepssm = list(protein_psepssm_file.columns)

R_all_train_test = "e_admat_dgc_mat_2_line.txt"
D_sim_file = "e_D_similarities.txt"
T_sim_file = "e_T_similarities.txt"


# In[7]:

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


# In[8]:

mode_list = ["T"]
seeds = [7771, 8367, 22, 1812, 4659]


protein_feature_list = [protein_psepssm_file]
molecule_feature_list = [molecule_fp2_file]
col_name = col_name_psepssm


# In[9]:

#labels = mat2vec(DT)
#pd.DataFrame(labels).to_csv('/home/dqw_cyy/11_MFDF_code/4_generate_dataset/E/E_labels.csv', index = False)


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

    for seed in seeds:
        folds_features,labels,test_idx = generate_10_cv_data(mode,seed,cv_data,DT,D_sim,T_sim,diDs,diTs,molecule_feature_list,protein_feature_list)
        i = 0
        for fold_data,test_idx_fold in zip(folds_features,test_idx):
            pd.DataFrame(fold_data).to_csv('/lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/4_generate_dataset/E/E_folddata_S' + str(mode) + '_seed' + str(seed) + '_fold' + str(i) + '.csv',index = False)
            pd.DataFrame(test_idx_fold).to_csv('/lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/4_generate_dataset/E/E_testidx_S' + str(mode) + '_seed' + str(seed) + '_fold' + str(i) + '.csv', index = False)
            i += 1

