
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy import signal

mode_list = ["p","D","T"]

def results_file(dataset = 'NR', mode = 'p',seeds = [7771, 8367, 22, 1812, 4659]):
    
    if dataset == 'NR':
        num_sample = 1404
    elif dataset == 'GPCR':
        num_sample = 21185
    elif dataset == 'IC':
        num_sample = 42840
    elif dataset == 'E':
        num_sample = 295480
    
    if mode == 'p':    
        temp = np.array([0.0] * num_sample * 488)
        temp = temp.reshape(num_sample,488) 
    else:
        temp = np.array([0.0] * num_sample * 486)
        temp = temp.reshape(num_sample,486) 
        
    for seed in seeds:
        print('----------------- mode: %s ,seed: %s' % (mode,seed),'-----------------')

        file_seq_folddata = glob.glob('./4_generate_dataset/' + dataset + '/' + dataset + '_folddata_S' + str(mode) + '_seed' + str(seed) + '_fold*' + '.csv')
        file_seq_folddata.sort()

        for i in range(10):
            print(file_seq_folddata[i])
            fold_data = pd.read_csv(file_seq_folddata[i])
            temp += np.array(fold_data)
    
    temp /= 50 # 10 * len(seeds)
    pd.DataFrame(temp).to_csv('./mean_' + dataset + '_folddata_S' + str(mode) + '.csv')
    return pd.DataFrame(temp)


def corr_results(data, mode = 'p'):
    print(data.shape[0])
    if mode == 'p':
        data_PathCS = pd.DataFrame(np.array(data.iloc[:,:12]).flatten())
        data_FP2 = pd.DataFrame(np.array(data.iloc[:,12:268]).flatten())
        data_PsePSSM = pd.DataFrame(np.array(data.iloc[:,268:]).flatten())
        assert (len(data_PathCS) == data.shape[0] * 12) # 16848 (NR)
        assert (len(data_FP2) == data.shape[0] * 256) # 359424 (NR)
        assert (len(data_PsePSSM) == data.shape[0] * 220) # 308880 (NR)
    else:
        data_PathCS = pd.DataFrame(np.array(data.iloc[:,:10]).flatten())
        data_FP2 = pd.DataFrame(np.array(data.iloc[:,10:266]).flatten())
        data_PsePSSM = pd.DataFrame(np.array(data.iloc[:,266:]).flatten())
        assert (len(data_PathCS) == data.shape[0] * 10)
        assert (len(data_FP2) == data.shape[0] * 256) 
        assert (len(data_PsePSSM) == data.shape[0] * 220)    
    
    #R_xx,R_yy
    mse_PathCS = signal.correlate(data_PathCS,data_PathCS, mode = 'valid')[0]
#    mse_FP2 = (data_FP2 ** 2).sum() / len(data_FP2)
    mse_PsePSSM = signal.correlate(data_PsePSSM,data_PsePSSM, mode = 'valid')[0]
    
    #R_xy
    R_PathCS_vs_FP2 = (signal.correlate(data_PathCS,data_FP2, mode = 'valid') / len(data_PathCS))
    R_PathCS_vs_PsePSSM = (signal.correlate(data_PathCS,data_PsePSSM, mode = 'valid') / len(data_PathCS))
    R_PsePSSM_vs_FP2 = (signal.correlate(data_PsePSSM,data_FP2, mode = 'valid') / len(data_PsePSSM))
    
    #corr_PathCS_vs_FP2
    corr_PathCS_vs_FP2 = []
    for i in range(0, len(R_PathCS_vs_FP2), data.shape[0]):
        y_FP2 = data_FP2[0][i:(i + len(data_PathCS))]
        R_FP2 = signal.correlate(y_FP2,y_FP2, mode = 'valid')[0]
        corr_PathCS_vs_FP2.append(R_PathCS_vs_FP2[i] / np.sqrt(mse_PathCS * R_FP2))
        if i % 100 == 0:
            print(i)
    print(abs(pd.DataFrame(corr_PathCS_vs_FP2)).max())
    print('===================corr_PathCS_vs_FP2======================')            
    #corr_PathCS_vs_PSSM
    corr_PathCS_vs_PsePSSM = []
    for i in range(0, len(R_PathCS_vs_PsePSSM), data.shape[0]):
        y_PsePSSM = data_PsePSSM[0][i:(i + len(data_PathCS))]
        R_PsePSSM = signal.correlate(y_PsePSSM,y_PsePSSM, mode = 'valid')[0]
        corr_PathCS_vs_PsePSSM.append(R_PathCS_vs_PsePSSM[i] / np.sqrt(mse_PathCS * R_PsePSSM))
        if i % 100 == 0:
            print(i)
    print(abs(pd.DataFrame(corr_PathCS_vs_PsePSSM)).max())
    print('===================corr_PathCS_vs_PsePSSM======================')               
    #corr_PSSM_vs_FP2
    corr_PsePSSM_vs_FP2 = []
    for i in range(0, len(R_PsePSSM_vs_FP2), data.shape[0]):
        y_FP2 = data_FP2[0][i:(i + len(data_PsePSSM))]
        R_FP2 = signal.correlate(y_FP2,y_FP2, mode = 'valid')[0]
        corr_PsePSSM_vs_FP2.append(R_PsePSSM_vs_FP2[i] / np.sqrt(mse_PsePSSM * R_FP2))        
        if i % 100 == 0:
            print(i)
    print(abs(pd.DataFrame(corr_PsePSSM_vs_FP2)).max())
    print('===================corr_PSSM_vs_FP2======================')              
    return pd.DataFrame(corr_PathCS_vs_FP2), pd.DataFrame(corr_PathCS_vs_PsePSSM), pd.DataFrame(corr_PsePSSM_vs_FP2)


def construct_corr_matrice(corr_PathCS_vs_FP2, corr_PathCS_vs_PSSM, corr_PSSM_vs_FP2):
    
    corr_matrice = pd.DataFrame([[1, abs(corr_PathCS_vs_FP2).max()[0], abs(corr_PathCS_vs_PSSM).max()[0]],
                                [abs(corr_PathCS_vs_FP2).max()[0], 1, abs(corr_PSSM_vs_FP2).max()[0]],
                                [abs(corr_PathCS_vs_PSSM).max()[0], abs(corr_PSSM_vs_FP2).max()[0], 1]])
    
    corr_matrice.rename(index = {0:'PathCS',1: 'FP2',2: 'PSSM'},
                        columns = {0:'PathCS',1: 'FP2',2: 'PSSM'},
                        inplace = True)
    
    return corr_matrice

# # NR
# # the code of GPCR, IC and E is the same as NR

NR_Sp_mean_data = results_file(dataset = 'NR',mode = 'p',seeds = [7771, 8367, 22, 1812, 4659])
NR_ST_mean_data = results_file(dataset = 'NR',mode = 'T',seeds = [7771, 8367, 22, 1812, 4659])
NR_SD_mean_data = results_file(dataset = 'NR',mode = 'D',seeds = [7771, 8367, 22, 1812, 4659])

NR_Sp_mean_data = pd.read_csv('mean_NR_folddata_Sp.csv',index_col = 0)
NR_corr_Sp_PathCS_vs_FP2, NR_corr_Sp_PathCS_vs_PSSM, NR_corr_Sp_PSSM_vs_FP2 = corr_results(NR_Sp_mean_data, mode = 'p')
NR_corr_Sp_matrice = construct_corr_matrice(NR_corr_Sp_PathCS_vs_FP2, NR_corr_Sp_PathCS_vs_PSSM, NR_corr_Sp_PSSM_vs_FP2)

NR_SD_mean_data = pd.read_csv('mean_NR_folddata_SD.csv',index_col = 0)
NR_corr_SD_PathCS_vs_FP2, NR_corr_SD_PathCS_vs_PSSM, NR_corr_SD_PSSM_vs_FP2 = corr_results(NR_SD_mean_data, mode = 'D')
NR_corr_SD_matrice = construct_corr_matrice(NR_corr_SD_PathCS_vs_FP2, NR_corr_SD_PathCS_vs_PSSM, NR_corr_SD_PSSM_vs_FP2)

NR_ST_mean_data = pd.read_csv('mean_NR_folddata_ST.csv',index_col = 0)
NR_corr_ST_PathCS_vs_FP2, NR_corr_ST_PathCS_vs_PSSM, NR_corr_ST_PSSM_vs_FP2 = corr_results(NR_ST_mean_data, mode = 'T')
NR_corr_ST_matrice = construct_corr_matrice(NR_corr_ST_PathCS_vs_FP2, NR_corr_ST_PathCS_vs_PSSM, NR_corr_ST_PSSM_vs_FP2)

NR_corr_ST_matrice.to_csv('NR_ST_corr_matrice.csv')
NR_corr_SD_matrice.to_csv('NR_SD_corr_matrice.csv')
NR_corr_Sp_matrice.to_csv('NR_Sp_corr_matrice.csv')