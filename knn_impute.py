#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:34:09 2019

@author: hamedniakan
"""

from fancyimpute import KNN
import pandas as pd 

data_train = pd.read_csv('/wsu/home/gn/gn85/gn8525/GP_SALES/data_train.csv' )
data_test = pd.read_csv('/wsu/home/gn/gn85/gn8525/GP_SALES/data_test.csv')


data_train_knn = pd.DataFrame(KNN(k=6).fit_transform(np.array(data_train.iloc[:,1:])) , columns = data_train.columns[1:])
data_train_knn['id'] = data_train['id']
cols = data_train_knn.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_train_knn = data_train_knn[cols] 
data_train_knn.to_csv('/wsu/home/gn/gn85/gn8525/GP_SALES/data_train_knn.csv')

data_test_knn = pd.DataFrame(KNN(k=6).fit_transform(np.array(data_test.iloc[:,1:])) , columns = data_test.columns[1:])
data_test_knn['id'] = data_test['id']
cols = data_test_knn.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_test_knn = data_test_knn[cols] 
data_test_knn.to_csv('/wsu/home/gn/gn85/gn8525/GP_SALES/data_test_knn.csv')