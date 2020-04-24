#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:35:19 2019

@author: hamedniakan
"""

import pandas as pd 
import numpy as np 
import re 
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
from matplotlib.gridspec import GridSpec
from fancyimpute import KNN
from sklearn.preprocessing import PolynomialFeatures , MinMaxScaler
from sklearn import linear_model
import multiprocessing as mp
from sklearn.cluster import KMeans
import time
from sklearn import preprocessing


data_dropped= pd.read_csv('/wsu/home/gn/gn85/gn8525/GP_SALES/GP_SE_regression/data_drop.csv')[['id','R27','U27','MV1','EXP1']]

data_dropped['month']=data_dropped['id'].apply(lambda x : int(x.split('_')[2]))
data_dropped['year']=data_dropped['id'].apply(lambda x : int(x.split('_')[1]))
data_dropped['dealer']=data_dropped['id'].apply(lambda x : (x.split('_')[0]))



def GP_SE(dealer):
    df = data_dropped[data_dropped['dealer'] == dealer]
    df.sort_values(['year', 'month' ] , inplace = True)
    # If a column is all nan , it would drop the feature so , the dimension wont be matched.
    try:
        
        df.iloc[:,1:]= SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(np.array(df.iloc[:,1:]))
       
    except:
        return [dealer]
        
    low_whisker = df.iloc[:,1:5].quantile(0.01)
    mask_1 = (df.iloc[:,1:5] < low_whisker)
    df.iloc[:,1:5] = np.where(mask_1 , low_whisker , df.iloc[:,1:5])
    
    high_whisker = df.iloc[:,1:5].quantile(0.99)
    mask_2 = (df.iloc[:,1:5] > high_whisker)
    df.iloc[:,1:5] = np.where(mask_2 , high_whisker , df.iloc[:,1:5])
    df['GPNV'] = df['R27']/df['U27']
    df['SE'] = df['MV1'] / df['EXP1']
    df = df.replace([np.inf , -np.inf] , np.nan)
    df.dropna(inplace = True)
    corr = df[['GPNV', 'SE']].corr().iloc[0,1]
    X_GPNV = MinMaxScaler().fit_transform(np.array(df['GPNV']).reshape(-1,1))
    y_SE = MinMaxScaler().fit_transform(np.array(df['SE']).reshape(-1,1))
    df['GPNV_scaled'] = X_GPNV
    df['SE_scaled'] = y_SE
    X_2 = PolynomialFeatures(degree =2) .fit_transform(X_GPNV)
    regress_2 = linear_model.LinearRegression()
    regress_2.fit(X_2, y_SE)
    X_3 = PolynomialFeatures(degree =3).fit_transform(X_GPNV)
    regress_3 = linear_model.LinearRegression()
    regress_3.fit(X_3, y_SE)
    k=KMeans(1)
    radius = max(k.fit_transform(df[['GPNV_scaled','SE_scaled']]))
    centroid = k.cluster_centers_
    collector =[dealer ,df.shape[0],  df['SE'].mean() , df['GPNV'].mean() , corr , radius , centroid[0] , centroid[1]]
    
    collector.extend(list(regress_2.coef_[0,:]))
    collector.append(regress_2.score(X_2 , np.array(df['SE'])))
    collector.extend(list(regress_3.coef_[0,:]))
    collector.append(regress_3.score(X_3 , np.array(df['SE'])))
    
    
    
    return df , collector


start = time.time()
pool = mp.Pool(mp.cpu_count())
results = pool.map(GP_SE ,[dealer for dealer in data_dropped['dealer'].unique()])
pool.close()

data_imputed = pd.DataFrame()
coef=[]
dealer_noinfo=[]
for tpl in results:
    if len(tpl) ==1:
        dealer_noinfo.append(tpl[0])
    else:
        data_imputed = pd.concat([data_imputed , tpl[0]] , axis = 0 , ignore_index= True)
        coef.append(tpl[1])
coef = pd.DataFrame(coef , 
                    columns=['dealer' ,'#ofrecords','SE_mean', 'GP_mean' , 'correlation', 'radius', 'centroid[0]',
                             'centroid[1]' , 'B0' , 'B1' , 'B2', 'R2_deg2' , 'C0', 'C1', 'C2' , 'C3', 'R2_deg3'])

    
    
duration = time.time()-start   


data_imputed.to_csv('~/data_sorted_imputed_regress_dealer_individually.csv' , index = False)
coef.to_csv('~/coef_dealer_individually.csv', index= False) 