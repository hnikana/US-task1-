#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:39:38 2019

@author: hamedniakan
"""


import pandas as pd
import numpy as np 

fca_distance = pd.read_csv('FCA_Distance_Append.txt')


rank = ['rank','industry_rank']

distance = ['distance', 'drivedistance']

wgt = ['WGT_outshr_ALL_10' , 'WGT_outshr_ALL_30' , 'WGT_outshr_COMpet1_10' , 'WGT_outshr_COMpet1_30' , 
    'WGT_outshr_COMpet2_10' , 'WGT_outshr_COMpet2_30' , 'outshr_ALL_10' , 'outshr_ALL_30' , 'outshr_COMpet1_10' ,
    'outshr_COMpet1_30' , 'outshr_COMpet2_10', 'outshr_COMpet2_30' ]

net = ['net_CLOSER_COMPET1SAME' , 'net_CLOSER_COMPET1p150' , 'net_CLOSER_COMPET2SAME' , 'net_CLOSER_COMPET2p150',
       'net_CLOSER_COMPETALLSAME' , 'net_CLOSER_COMPETALLp150' , 
       'net_CLOSER_WGTCOMPET1SAME' , 'net_CLOSER_WGTCOMPET1p150' , 
       'net_CLOSER_WGTCOMPET2SAME' , 'net_CLOSER_WGTCOMPET2p150' , 
       'net_CLOSER_WGTCOMPETALLSAME' , 'net_CLOSER_WGTCOMPETALLp150' ]  


# def wavg(group, avg_name, weight_name):
#     # import pdb; pdb.set_trace()
#     """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
#     In rare instance, we may not have weights, so just return the mean. Customize this if your business case
#     should return otherwise.
#     """
#     d = group[avg_name]
#     w = group[weight_name]
#     try:
#         return (d * w).median() / w.sum()
#     except ZeroDivisionError:
#         return d.mean()
    
    

    
    
# wtavg = lambda x: np.median(x[rank] *  x['volume'], axis = 0)



df = fca_distance[rank + distance + wgt + net].multiply(fca_distance['volume'] , axis = 0 )
df[['dealer' , 'volume']] = fca_distance[['dealer' , 'volume']]

df_agg = df.groupby('dealer').apply(lambda x : x[rank + distance + wgt + net].median() / x['volume'].sum()).reset_index()

df_agg.to_pickle('fca_distance_agg.pkl')



