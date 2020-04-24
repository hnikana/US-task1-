#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:52:26 2019

@author: hamedniakan
"""

import pandas as pd 
import numpy as np 
from fancyimpute import KNN
import multiprocessing as mp
import time



dataset_acc = pd.read_csv('Proj_2_export/dataset_acc.csv' )
zero_frequency = pd.DataFrame((dataset_acc == 0).sum().reset_index(name = 'frequency'))
zero_frequency['frequency'] = zero_frequency['frequency']/ dataset_acc.shape[0]
missing_freq = dataset_acc.isna().sum().reset_index(name = 'frequency')
missing_freq['frequency'] = missing_freq['frequency']/dataset_acc.shape[0]
zero_missing_freq = pd.merge(missing_freq , zero_frequency , on = 'index' , how = 'left' )
zero_missing_freq ['sum'] = zero_missing_freq['frequency_x']+zero_missing_freq['frequency_y']
columns_to_drop = list(zero_missing_freq[zero_missing_freq['sum']> 0.1]['index'] )
data_dropped = dataset_acc.drop(columns = columns_to_drop)

#data_dropped= pd.read_csv('./Proj_2_export/export_csv/data_drop.csv')[['id','R27','U27','MV1','EXP1']]

data_dropped['month']=data_dropped['id'].apply(lambda x : int(x.split('_')[2]))
data_dropped['year']=data_dropped['id'].apply(lambda x : int(x.split('_')[1]))
data_dropped['dealer']=data_dropped['id'].apply(lambda x : (x.split('_')[0]))



''' 
!!! Dr.chinnam  asked to just keep the features which comntribute in constructing GPNV and SE. 
 However, at first I tried to keep all the features but , what happend was during the imputation some features for a specific
 was all nan values and it intrupts the program . 
 I could keep those features by a for loop and if statement but for time being I just work on features which contributes 
 to constructing the KPIs of our interest 
 in taht case is not very sensible to use knn because we have just dropeed many features that KNN could get benefit from 
 
 
def mask_outlier'''    
    
      
def account_imputing(dealer):
    df = data_dropped[data_dropped['dealer'] == dealer]
    df.sort_values(['year', 'month' ] , inplace = True)
    df.iloc[:,1:51] = KNN(k=5).fit_transform(np.array(df.iloc[:,1:51]))    
    low_whisker = df.iloc[:,1:51].quantile(0.01)
    high_whisker = df.iloc[:,1:51].quantile(0.99)
    mask_1 = (df.iloc[:,1:51] < low_whisker)
    mask_2 = (df.iloc[:,1:51] > high_whisker)
    df.iloc[:,1:51] = np.where(mask_1 , low_whisker , df.iloc[:,1:51])
    df.iloc[:,1:51] = np.where(mask_2 , high_whisker , df.iloc[:,1:51])
    
    return df


start = time.time()
pool = mp.Pool(mp.cpu_count())
results = pool.map(account_imputing ,[dealer for dealer in data_dropped['dealer'].unique()[:10]])
pool.close()

acc = pd.DataFrame()
for df in results:
    acc = pd.concat([acc , df] , axis = 0 , ignore_index= True)
        
duration = time.time()-start    