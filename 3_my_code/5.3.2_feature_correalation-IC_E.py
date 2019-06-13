
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy import signal


# # def：互相关系数

# In[7]:

def corr_results(data, mode = 'p'):
    
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
    for i in range(len(R_PathCS_vs_FP2)):
        try:
            signal.signal(signal.SIGALRM, handle)
            signal.alarm(1800)
            y_FP2 = data_FP2[0][i:(i + len(data_PathCS))]
            R_FP2 = signal.correlate(y_FP2,y_FP2, mode = 'valid')[0]
            corr_PathCS_vs_FP2.append(R_PathCS_vs_FP2[i] / np.sqrt(mse_PathCS * R_FP2))
            signal.alarm(0)
        except:
            print('超时:',i)
            continue

        if i % 10000 == 0:
            print(i)
    print(abs(pd.DataFrame(corr_PathCS_vs_FP2)).max())
    print('===================corr_PathCS_vs_FP2======================')  

    #corr_PathCS_vs_PSSM
    corr_PathCS_vs_PsePSSM = []
    for i in range(len(R_PathCS_vs_PsePSSM)):
        try:
            signal.signal(signal.SIGALRM, handle)
            signal.alarm(1800)
            y_PsePSSM = data_PsePSSM[0][i:(i + len(data_PathCS))]
            R_PsePSSM = signal.correlate(y_PsePSSM,y_PsePSSM, mode = 'valid', method = 'fft')[0]
            corr_PathCS_vs_PsePSSM.append(R_PathCS_vs_PsePSSM[i] / np.sqrt(mse_PathCS * R_PsePSSM))
            signal.alarm(0)
        except:
            print('超时:',i)
            continue

        if i % 10000 == 0:
            print(i)
    print(abs(pd.DataFrame(corr_PathCS_vs_PsePSSM)).max())
    print('===================corr_PathCS_vs_PsePSSM======================')               

    #corr_PSSM_vs_FP2
    corr_PsePSSM_vs_FP2 = []
    for i in range(len(R_PsePSSM_vs_FP2)):
        try:
            signal.signal(signal.SIGALRM, handle)
            signal.alarm(1800)
            y_FP2 = data_FP2[0][i:(i + len(data_PsePSSM))]
            R_FP2 = signal.correlate(y_FP2,y_FP2, mode = 'valid')[0]
            corr_PsePSSM_vs_FP2.append(R_PsePSSM_vs_FP2[i] / np.sqrt(mse_PsePSSM * R_FP2))
            signal.alarm(0)
        except:
            print('超时:',i)
            continue

        if i % 10000 == 0:
            print(i)
    print(abs(pd.DataFrame(corr_PsePSSM_vs_FP2)).max())
    print('===================corr_PSSM_vs_FP2======================')              
    return pd.DataFrame(corr_PathCS_vs_FP2), pd.DataFrame(corr_PathCS_vs_PsePSSM), pd.DataFrame(corr_PsePSSM_vs_FP2)

# # 构建：相关系数矩阵

# In[25]:

def construct_corr_matrice(corr_PathCS_vs_FP2, corr_PathCS_vs_PSSM, corr_PSSM_vs_FP2):
    
    corr_matrice = pd.DataFrame([[1, abs(corr_PathCS_vs_FP2).max()[0], abs(corr_PathCS_vs_PSSM).max()[0]],
                                [abs(corr_PathCS_vs_FP2).max()[0], 1, abs(corr_PSSM_vs_FP2).max()[0]],
                                [abs(corr_PathCS_vs_PSSM).max()[0], abs(corr_PSSM_vs_FP2).max()[0], 1]])
    
    corr_matrice.rename(index = {0:'PathCS',1: 'FP2',2: 'PSSM'},
                        columns = {0:'PathCS',1: 'FP2',2: 'PSSM'},
                        inplace = True)
    
    return corr_matrice


# # 画图
#     E:\11_MFDF_code\4_generate_dataset\NR\NR_folddata_Sp_seed7771_fold9.csv

# In[4]:

def draw_corr_matrice(corr_matrice, dataset = 'NR', mode = 'p'):
    sns.set()
    plt.rcParams['figure.dpi'] = 300
    
    sns.heatmap(corr_matrice, cmap=plt.cm.summer, annot=True)
    
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.xaxis.set_ticks_position('top')
    ax.spines['top'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    plt.text(1.5 ,3.3,dataset + '_S' + mode + '_Correlation between three features',verticalalignment = 'bottom', horizontalalignment = 'center')
    plt.savefig('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/' + dataset + '/' + dataset + '_S' + mode + '_corr_matrice_(2).png', dpi = 300)
    #plt.show()
    return 


# # NR_求解：互相关系数 

# In[8]:

#NR_Sp_mean_data = pd.read_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/5.4_mean_NR_folddata_Sp.csv',index_col = 0)
#NR_corr_Sp_PathCS_vs_FP2, NR_corr_Sp_PathCS_vs_PSSM, NR_corr_Sp_PSSM_vs_FP2 = corr_results(NR_Sp_mean_data, mode = 'p')
#NR_corr_Sp_PathCS_vs_FP2, NR_corr_Sp_PathCS_vs_PSSM, NR_corr_Sp_PSSM_vs_FP2
#
#
## In[27]:
#
#NR_SD_mean_data = pd.read_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/5.4_mean_NR_folddata_SD.csv',index_col = 0)
#NR_corr_SD_PathCS_vs_FP2, NR_corr_SD_PathCS_vs_PSSM, NR_corr_SD_PSSM_vs_FP2 = corr_results(NR_SD_mean_data, mode = 'D')
#NR_corr_SD_PathCS_vs_FP2, NR_corr_SD_PathCS_vs_PSSM, NR_corr_SD_PSSM_vs_FP2
#
#
## In[28]:
#
#NR_ST_mean_data = pd.read_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/5.4_mean_NR_folddata_ST.csv',index_col = 0)
#NR_corr_ST_PathCS_vs_FP2, NR_corr_ST_PathCS_vs_PSSM, NR_corr_ST_PSSM_vs_FP2 = corr_results(NR_ST_mean_data, mode = 'T')
#NR_corr_ST_PathCS_vs_FP2, NR_corr_ST_PathCS_vs_PSSM, NR_corr_ST_PSSM_vs_FP2
#
#
## # NR_结果
#
## In[29]:
#
#NR_corr_ST_matrice = construct_corr_matrice(NR_corr_ST_PathCS_vs_FP2, NR_corr_ST_PathCS_vs_PSSM, NR_corr_ST_PSSM_vs_FP2)
#draw_corr_matrice(NR_corr_ST_matrice, dataset = 'NR', mode = 'T')
#NR_corr_ST_matrice
#
#
## In[26]:
#
#NR_corr_Sp_matrice = construct_corr_matrice(NR_corr_Sp_PathCS_vs_FP2, NR_corr_Sp_PathCS_vs_PSSM, NR_corr_Sp_PSSM_vs_FP2)
#draw_corr_matrice(NR_corr_Sp_matrice, dataset = 'NR', mode = 'p')
#NR_corr_Sp_matrice
#
#
## In[30]:
#
#NR_corr_SD_matrice = construct_corr_matrice(NR_corr_SD_PathCS_vs_FP2, NR_corr_SD_PathCS_vs_PSSM, NR_corr_SD_PSSM_vs_FP2)
#draw_corr_matrice(NR_corr_SD_matrice, dataset = 'NR', mode = 'D')
#NR_corr_SD_matrice


# # GPCR_结果

## In[22]:
#
#GPCR_Sp_mean_data = pd.read_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/5.4_mean_GPCR_folddata_Sp.csv',index_col = 0)
#GPCR_corr_Sp_PathCS_vs_FP2, GPCR_corr_Sp_PathCS_vs_PSSM, GPCR_corr_Sp_PSSM_vs_FP2 = corr_results(GPCR_Sp_mean_data, mode = 'p')
#GPCR_corr_Sp_PathCS_vs_FP2, GPCR_corr_Sp_PathCS_vs_PSSM, GPCR_corr_Sp_PSSM_vs_FP2
#
#
## In[71]:
#
#GPCR_corr_Sp_matrice = construct_corr_matrice(GPCR_corr_Sp_PathCS_vs_FP2, GPCR_corr_Sp_PathCS_vs_PSSM, GPCR_corr_Sp_PSSM_vs_FP2)
#draw_corr_matrice(GPCR_corr_Sp_matrice, dataset = 'GPCR', mode = 'p')
#GPCR_corr_Sp_matrice
#
#
## In[23]:
#
#GPCR_SD_mean_data = pd.read_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/5.4_mean_GPCR_folddata_SD.csv',index_col = 0)
#GPCR_corr_SD_PathCS_vs_FP2, GPCR_corr_SD_PathCS_vs_PSSM, GPCR_corr_SD_PSSM_vs_FP2 = corr_results(GPCR_SD_mean_data, mode = 'D')
#
#
## In[73]:
#
#GPCR_corr_SD_matrice = construct_corr_matrice(GPCR_corr_SD_PathCS_vs_FP2, GPCR_corr_SD_PathCS_vs_PSSM, GPCR_corr_SD_PSSM_vs_FP2)
#draw_corr_matrice(GPCR_corr_SD_matrice, dataset = 'GPCR', mode = 'D')
#GPCR_corr_SD_matrice
#
#
## In[24]:
#
#GPCR_ST_mean_data = pd.read_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/5.4_mean_GPCR_folddata_ST.csv',index_col = 0)
#GPCR_corr_ST_PathCS_vs_FP2, GPCR_corr_ST_PathCS_vs_PSSM, GPCR_corr_ST_PSSM_vs_FP2 = corr_results(GPCR_ST_mean_data, mode = 'T')
#
#
## In[75]:
#
#GPCR_corr_ST_matrice = construct_corr_matrice(GPCR_corr_ST_PathCS_vs_FP2, GPCR_corr_ST_PathCS_vs_PSSM, GPCR_corr_ST_PSSM_vs_FP2)
#draw_corr_matrice(GPCR_corr_ST_matrice, dataset = 'GPCR', mode = 'T')
#GPCR_corr_ST_matrice


# # 结果：IC

# In[26]:
print('====================================IC_ST========================')
IC_ST_mean_data = pd.read_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/5.4_mean_IC_folddata_ST.csv',index_col = 0)
IC_corr_ST_PathCS_vs_FP2, IC_corr_ST_PathCS_vs_PSSM, IC_corr_ST_PSSM_vs_FP2 = corr_results(IC_ST_mean_data, mode = 'T')

IC_corr_ST_matrice = construct_corr_matrice(IC_corr_ST_PathCS_vs_FP2, IC_corr_ST_PathCS_vs_PSSM, IC_corr_ST_PSSM_vs_FP2)
draw_corr_matrice(IC_corr_ST_matrice, dataset = 'IC', mode = 'T')
IC_corr_ST_matrice


# In[27]:
print('====================================IC_SD========================')
IC_SD_mean_data = pd.read_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/5.4_mean_IC_folddata_SD.csv',index_col = 0)
IC_corr_SD_PathCS_vs_FP2, IC_corr_SD_PathCS_vs_PSSM, IC_corr_SD_PSSM_vs_FP2 = corr_results(IC_SD_mean_data, mode = 'D')

IC_corr_SD_matrice = construct_corr_matrice(IC_corr_SD_PathCS_vs_FP2, IC_corr_SD_PathCS_vs_PSSM, IC_corr_SD_PSSM_vs_FP2)
draw_corr_matrice(IC_corr_SD_matrice, dataset = 'IC', mode = 'D')
IC_corr_SD_matrice


# In[28]:
print('====================================IC_SP========================')
IC_Sp_mean_data = pd.read_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/5.4_mean_IC_folddata_Sp.csv',index_col = 0)
IC_corr_Sp_PathCS_vs_FP2, IC_corr_Sp_PathCS_vs_PSSM, IC_corr_Sp_PSSM_vs_FP2 = corr_results(IC_Sp_mean_data, mode = 'p')

IC_corr_Sp_matrice = construct_corr_matrice(IC_corr_Sp_PathCS_vs_FP2, IC_corr_Sp_PathCS_vs_PSSM, IC_corr_Sp_PSSM_vs_FP2)
draw_corr_matrice(IC_corr_Sp_matrice, dataset = 'IC', mode = 'p')
IC_corr_Sp_matrice


# # 结果：E

# In[29]:
print('====================================E_ST========================')
E_ST_mean_data = pd.read_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/5.4_mean_E_folddata_ST.csv',index_col = 0)
E_corr_ST_PathCS_vs_FP2, E_corr_ST_PathCS_vs_PSSM, E_corr_ST_PSSM_vs_FP2 = corr_results(E_ST_mean_data, mode = 'T')

E_corr_ST_matrice = construct_corr_matrice(E_corr_ST_PathCS_vs_FP2, E_corr_ST_PathCS_vs_PSSM, E_corr_ST_PSSM_vs_FP2)
draw_corr_matrice(E_corr_ST_matrice, dataset = 'E', mode = 'T')
E_corr_ST_matrice


# In[30]:
print('====================================E_SD========================')
E_SD_mean_data = pd.read_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/5.4_mean_E_folddata_SD.csv',index_col = 0)
E_corr_SD_PathCS_vs_FP2, E_corr_SD_PathCS_vs_PSSM, E_corr_SD_PSSM_vs_FP2 = corr_results(E_SD_mean_data, mode = 'D')

E_corr_SD_matrice = construct_corr_matrice(E_corr_SD_PathCS_vs_FP2, E_corr_SD_PathCS_vs_PSSM, E_corr_SD_PSSM_vs_FP2)
draw_corr_matrice(E_corr_SD_matrice, dataset = 'E', mode = 'D')
E_corr_SD_matrice


# In[31]:
print('====================================E_SP========================')
E_Sp_mean_data = pd.read_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/5.4_mean_E_folddata_Sp.csv',index_col = 0)
E_corr_Sp_PathCS_vs_FP2, E_corr_Sp_PathCS_vs_PSSM, E_corr_Sp_PSSM_vs_FP2 = corr_results(E_Sp_mean_data, mode = 'p')

E_corr_Sp_matrice = construct_corr_matrice(E_corr_Sp_PathCS_vs_FP2, E_corr_Sp_PathCS_vs_PSSM, E_corr_Sp_PSSM_vs_FP2)
draw_corr_matrice(E_corr_Sp_matrice, dataset = 'E', mode = 'p')
E_corr_Sp_matrice


# # 结果保存

# In[82]:

#corr_ST_matrice.to_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/NR_ST_corr_matrice.csv')
#corr_SD_matrice.to_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/NR_SD_corr_matrice.csv')
#corr_Sp_matrice.to_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/NR_Sp_corr_matrice.csv')
#GPCR_corr_ST_matrice.to_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/GPCR_ST_corr_matrice.csv')
#GPCR_corr_SD_matrice.to_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/GPCR_SD_corr_matrice.csv')
#GPCR_corr_Sp_matrice.to_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/GPCR_Sp_corr_matrice.csv')
IC_corr_ST_matrice.to_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/IC_ST_corr_matrice.csv')
IC_corr_SD_matrice.to_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/IC_SD_corr_matrice.csv')
IC_corr_Sp_matrice.to_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/IC_Sp_corr_matrice.csv')


# In[ ]:

E_corr_ST_matrice.to_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/E_ST_corr_matrice.csv')
E_corr_SD_matrice.to_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/E_SD_corr_matrice.csv')
E_corr_Sp_matrice.to_csv('/home/dqw_cyy/11_MFDF_code/5_feature_analysis/E_Sp_corr_matrice.csv')


