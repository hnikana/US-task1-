#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:26:58 2019

@author: hamedniakan
"""
#***************************
#data == acc level 
#KPI == kpi level 

#**************************


###############################################################################################
# Functions for constructiong KPIs from ACC info
###############################################################################################


'''
Files :
    KPI_formulas.csv : KPI names and corresponding formula 
'''
import pandas as pd 
import numpy as np 
import re 
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
from matplotlib.gridspec import GridSpec


# write a code to extract kpi name if exists in a sheet and extract kpi name and it calculation 
formula = pd.read_excel('./FS_Accts_Formulas_ExpSls_Inv_modified.xlsx' , sheet_name = None)

KPI_formula = pd.DataFrame()
for key in formula:
    if 'KPI Name' in formula[key].columns:
        KPI_formula = KPI_formula.append(formula[key][['KPI Name', 'Calculation']])
KPI_formula.to_csv('./Proj_2_export/KPI_formulas.csv' , index = False)        

'''
it takes a formula which is a string and also a dataset , 
it returns an array resulting from applying taht formula on corresponding columns of the dataset 
'''

        
def KPI_construction (dataset , formula):
    formula = formula.replace(' ','')
    formula = re.sub('[]()[]','',formula)
    formula = formula.split('/')
    for i in range(len(formula)) :
# it would be better to check whether it has more than one element or not like th efunction above         
        formula[i] = formula[i].split('+')
        #formula[i]= [j.strip() for j in formula[i]]
        if len(set(formula[i])-set(dataset.columns))==0 :            
            formula[i] = pd.DataFrame(dataset[formula[i]].sum(axis =1))
        else: 
            return None
    
    return formula[0]/formula[1]


    

def KPI_data(dataset , formula):
    df = pd.DataFrame(dataset.iloc[:,0])
    for i in range(formula.shape[0]):
       df[formula.iloc[i,0]]= KPI_construction(dataset, formula.iloc[i,1])
       
    return df 




'''
 second way of constructiong kpis # these line of codes trying to use parser function and formulas module 
 
This function works well nut it just taking more time and it s slow. It needs to be optimized , it would be useful if there are any other
    Operation rather than + or / . Formula modula could translate any operation based on excel format tp a function . 
The probklem with the function below is passing its argument . 
and because apply function is a row by row operation is too sloaw . 
I our KPIs we just have + and / , Manually I checked the values and it should be fine. 
So ,we do nt use the function below and module formula 


#############################
import formulas

df = pd.DataFrame(dataset.iloc[:,0]) 

for i in range(formula.shape[0]):
    f = re.sub('[]()[]','',formula.iloc[i,1]) # it could be applied on whole dataset first 
    f = '='+f
    func = formulas.Parser().ast(f)[1].compile()
    inputs  = list(func.inputs)
    if len(set(inputs)-set(dataset.columns))==0:
        #*args = (dataset[j][i] for j in inputs)
        df[formula.iloc[i,0]] = dataset[inputs].apply(lambda row: func(*(row[j]  for j in inputs)) , axis = 1 )
    else :
        df[formula.iloc[i,0]] = None #np.nan
        
##############################

'''
##############################################################################################################
# Removing the outlier Function 
##############################################################################################################
from pandas.api.types import is_numeric_dtype

def remove_outlier(df):
    low = .05
    high = .95
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
    return df

##############################################################################################################
# Reading data and collectiong account info with defining a unique id for each dealr ; 'dealer_year_month '
##############################################################################################################

'''
Export Files to get the account level dataset:
    data_concat.csv : all the data from three files provided concated together 
    data_concat_ID.csv : another colimn ID is the text column of dealer_Yr_Mth , in order to pivot over DispSymbol and the value of MTDAmount 
    data_concat_test_2019 : data derived from data_concat_Id.csv for test set before pivoting arount id 
    data_concat_train_2019below.csv : data derived from data_concat_Id.csv for train set before pivoting arount id
    trainset_acc.csv : Pivoting table derive from concat_train around id to get the feature space at the account level 
    testset_acc.csv : Pivoting derive from concat_test around id to get the feature space at the account level
    trainset_acc_dropexp3andexp23.csv : dropping exp3 and exp23 which does not exist in testset but in trainset
    dataset_acc.csv : whole data affte concatenation the two bove sets just acc level  
    
'''
'''
Reading Data 
    Train dataset , records related to 2019 and older 
    Test dataset , records related to 2019 

    pd.Pivot cannot be used beacuase of loosing the yr and month , however if I use multiple indecies (dealer , yr , month ) ; I could get the values
        but , I would have many columns 
    
    I used pd.get_dummies , but in order to not have conflict with next steps , constructing KPIs , prefix and prefix_ sep should equal to '' .
        Also , after we got the matrix sparse we should multiply matrix by the value (MTDamount) ; but it does not work why ? because the 
        dealers are not unique and for each dealer we will end up having multiple record (the feature space for each record is one hot vector )
        which we should gropby them together and sum over their features (That might work ) , 
    
    The other way is to creat a column with the format of (dealer_year_ month ) to make it unique and then pivot it over this coloumn and the value 
        is YTD amount and after we get the pivot table , we again sparse the dealer_yr_month into 3 columns based on necessity 
'''


data_financial = pd.read_csv('./Project2_Data/CDJR_Financial_Data.txt')
data_inventory = pd.read_csv('./Project2_Data/CDJR_Inventory_Data_2.txt')
data_Saleseffectiveness = pd.read_csv('./Project2_Data/CDJR_SlsEff_data_2.txt')

data_concat = pd.concat([data_financial, data_inventory , data_Saleseffectiveness] , ignore_index = True)
data_concat.to_csv('./Proj_2_export/data_concat.csv' , index = False)

data_concat['id'] = data_concat[['dealer','Yr', 'Mth']].apply(lambda x : '_'.join(x.map(str)), axis = 1)
data_concat.to_csv('./Proj_2_export/data_concat_ID.csv' , index = False)



data_concat_train = data_concat[data_concat.Yr<2019]
data_concat_train.to_csv('./Proj_2_export/data_concat_train_2019below.csv' , index = False)

data_concat_test = data_concat[data_concat.Yr==2019]
data_concat_test.to_csv('./Proj_2_export/data_concat_test_2019.csv' , index = False)

#data_concat_dumm = pd.get_dummies(data_concat , columns=['DispSymbol'] , prefix='', prefix_sep='')

'''
number of keys in excel file 286 
number of keys in data_concat_dumm = 268
'''

trainset = data_concat_train.pivot(index='id', columns='DispSymbol', values='MTDAmount').reset_index()
trainset.to_csv('./Proj_2_export/trainset_acc.csv' , index = False)

testset = data_concat_test.pivot(index='id', columns='DispSymbol', values='MTDAmount').reset_index()
testset.to_csv('./Proj_2_export/testset_acc.csv' , index = False)

assert (trainset.shape[0]+ testset.shape[0] == len(data_concat.id.unique())) , 'size are not the same'

assert (len(trainset.columns) == len(testset.columns)) , 'There is an inconsistency between train and test sets'

inconsistency = set(trainset.columns).difference(testset.columns)
   
''' 
!!! the EXP23 and EXP3 are the missing columns in testset (data related to 2019) and are related to CJDRStandardMIdsizeCar and CJDRCompactCar
I just drop them in test set as well 
KPIs which will be affected ;   
sheet 5 , record 3		Sales Effectiveness - CJDRCompactCar	MV3 / EXP3
sheet 5 record 23		Sales Effectiveness - CJDRStandardMidSizeCar	MV23 / EXP23
sheet 6 record 3		Inventory Per Expected - CJDRCompactCar	INV3 / EXP3        
sheet 6 record 23		Inventory Per Expected - CJDRStandardMidSizeCar	INV23 / EXP23 
Those KPIs have been dropped as well from excel formulas , formulas will be read from modified file. 
'''
trainset.drop(['EXP3' , 'EXP23'] , inplace = True , axis = 1)
trainset.to_csv('./Proj_2_export/trainset_acc_dropexp3andexp23.csv' , index = False)


dataset_acc = pd.concat([trainset_acc_dropexp3andexp23 , testset_acc] , ignore_index = True)
dataset_acc.to_csv('Proj_2_export/dataset_acc.csv' , index = False)





#################################################################################################
# ACCOUNT LEVEL DATA IMPUTATION PREPROCESSING 
#################################################################################################
'''
Exported CSV : 
    data_zero_frequency.csv : the frequency of zero values in columns
    data_missingvalue_frequency.csv
    data_drop.csv : dropping columns with more than 30 percent of having misiisng value or zero values
    data_train.csv : derived from data_drop.csv
        this is the acc level training data (2019 and less) in which the dealers with more than 30% zero+missing have been dropped 
    data_test.csv: ,,,,,
    data_train_imp.csv : mean imputation 
    data_test_imp.csv : mean imputation 
    kpi_train.csv : kpi construction from data_train_imp.csv
    kpi_test.csv :  kpi construction from data_test_imp.csv
    kpi_all.csv : concatenating kpi_train.csv & kpi_test.csv 
    kpi_all_drop.csv : dropping columns with more than 30 percent nan.values
    kpi_train_imp.csv : splitted at the index they joined before , from the kpi_all_drop, records with motre than 30 percent
        of nan , inf and zero dropped then all imputed by zero , 
    kpi_train_imp.csv : ,,,,
    
    
'''
'''
Exported graph : 
    data_zero_frequency.png : the frequency of zero values per column
    data_missingvalue_frequency.png
    data_zero_missing_frequency.png
    data_histogram_records_invalidvalues.png
    kpi_all_invalidvalues.png
    kpi_all_invalidvalues_afterdrop.png
    kpi_histogram_records_invalidvalues.png
    
    
'''



'''
    1- zero valu analysis per column 
    2- missing value analysis 
    3- combination of zero and missing and droping the values with more than 30 percent
    4- droping the rows with more than 30% zero values and missing values after spliting to train and test to keep track of 
        shape of train and test 
    5- Imputation by mean (KNN imputer should be try on grid )
    6- constructing the KPIs for each of them 
    7- concatenate the kpis to analyze the invalid values 
    8- Analysis of INF , NAN and Zero  at KPI level
    9- droping kpis with more than 30 percent of those invalid vals 
    10- analyize the invalid vals per row after spliting to test and train 
    11-droping rows from tarin and test with more than 30 percent of invalid values 
    12- impute np.inf and np.nan with zero 
    
    
'''



zero_frequency = pd.DataFrame((dataset_acc == 0).sum().reset_index(name = 'frequency'))
zero_frequency['frequency'] = zero_frequency['frequency']/ dataset_acc.shape[0]
zero_frequency.to_csv('Proj_2_export/export_csv/data_zero_frequency.csv' , index = False) 

fig = plt.figure(figsize=(12,8))
color = ['blue' if x <.3 else 'red' for x in zero_frequency['frequency']]
plt.bar(range(zero_frequency.shape[0]) , zero_frequency['frequency'] , color = color )
plt.title('Frequency of zero values_Account level' , loc = 'center')
plt.xlabel('ACC_index')
plt.ylabel('Percentage')
fig.savefig('./Proj_2_export/export_graph/data_zero_frequency.png')



# Analysis of missing values at acc level and figures 
missing_freq = dataset_acc.isna().sum().reset_index(name = 'frequency')
missing_freq['frequency'] = missing_freq['frequency']/dataset_acc.shape[0]
missing_freq.to_csv('Proj_2_export/export_csv/data_missingvalue_frequency.csv' , index = False)

fig = plt.figure(figsize=(12,8))
color = ['blue' if x <.3 else 'red' for x in missing_freq['frequency']]
plt.bar(range(missing_freq.shape[0]) , missing_freq['frequency'] , color = color )
plt.title('Frequency of Missing VAlues_Account level' , loc = 'center')
plt.xlabel('ACC_index')
plt.ylabel('Percentage')
fig.savefig('./Proj_2_export/export_graph/data_missingvalue_frequency.png')

'''
!!! The graph is showing that the amount of missing data for many of features are the same meaning a portion of dealers have the same amount of 
    missing values , most probably the same ones 
'''

# zero + missing freq 
zero_missing_freq = pd.merge(missing_freq , zero_frequency , on = 'index' , how = 'left' )
zero_missing_freq ['sum'] = zero_missing_freq['frequency_x']+zero_missing_freq['frequency_y']

fig = plt.figure(figsize=(12,8))
color = ['blue' if x <.3 else 'red' for x in zero_missing_freq['sum']]
plt.bar(range(zero_missing_freq.shape[0]) , zero_missing_freq['sum'] , color = color )
plt.title('Frequency of zero values and missing values_Account level' , loc = 'center')
plt.xlabel('ACC_index')
plt.ylabel('Percentage')
fig.savefig('./Proj_2_export/export_graph/data_zero_missing_frequency.png')

#droping columns with more than 30 percent missing values + zero values 
# !!! from 267 it dropped to 110 
columns_to_drop = list(zero_missing_freq[zero_missing_freq['sum']> 0.3]['index'] )
data_dropped = dataset_acc.drop(columns = columns_to_drop)
data_dropped.to_csv('./Proj_2_export/export_csv/data_drop.csv' , index = False)



#--------------------------------------------------------------------------------------------------------
# Analyze of zeros to see whether they are missing or really zero
#zero values analysis 
#   In this part I tries to see whther zeros are really zero or missing 
#   Box plot is showing the box plot of each features, note that they are not dealer rep[resentation (It is besed
#   on dealer-yr-month)
#   Analyze whteher zeros are true values or just missing ==> before dropping any columns except those are not in 
#   Testset 
#   I - Visualizations
#--------------------------------------------------------------------------------------------------------
dataset_acc = pd.read_csv('Proj_2_export/dataset_acc.csv')

stat_1 = dataset_acc.quantile([0,.001,.01 , .1,.5,.9 , .95 , .99, .999,1])


fig = plt.figure(figsize = (20,100))
fig.suptitle('Feature Box plot _ accounts ')
gs=GridSpec(27,1 )

for i in range(27):
    plt.subplot(gs[i,0])
    dataset_acc.iloc[:,i*10:(i+1)*10].boxplot(vert = False)

fig.tight_layout()
fig.savefig('./Proj_2_export/export_graph/accounts_box_plot.png' , dpi = 200)

for j in stat_1.columns :
    stat_1.loc['Med',j] = dataset_acc.loc[dataset_acc[j].to_numpy().nonzero()[0],j].median()


stat_1.loc['alpha',:] = np.array(zero_frequency.iloc[1:,1] )/ (1- np.array(missing_freq.iloc[1:,1]))
stat_1.loc['shift',:] = (stat_1.loc['Med',:]-stat_1.loc[.5,:])
#--------------------------------------------------------------------------------------------------------
 
data_dropped = pd.read_csv('./Proj_2_export/export_csv/data_drop.csv')

fig = plt.figure(figsize = (20,100))
fig.suptitle('Feature Box plot _ accounts ')
gs=GridSpec(11,1 )

for i in range(11):
    plt.subplot(gs[i,0])
    data_dropped.iloc[:,i*10:(i+1)*10].boxplot(vert = False)

fig.tight_layout()
fig.savefig('./Proj_2_export/export_graph/accounts_afterdropzero&missing_box_plot.png' , dpi = 200)



# In order to drop the records with m ore than 30 percent of missing and zero values , I need to split them up to train and test
# Because, I could keep the track of shape , if I drop them here , I need to figure out AGAIN what record is 2019 or below  
data_train = data_dropped.iloc[:63360,:].copy()
data_test = data_dropped.iloc[63360:,:].copy() 

# droping training records with high missing values or zeros at rows , records drop to 57022 from 63360 == 6338 ; 10% 
# droping training records with high missing values or zeros at rows , records drop to 4588 from 5280 == 692 ; 13 % 

data_train['freq'] = (data_train.isna().sum(axis=1) + (data_train == 0).sum(axis =1 ))/ data_train.shape[1]
data_test['freq'] = (data_test.isna().sum(axis=1) + (data_test == 0).sum(axis =1 ))/ data_test.shape[1]

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
ax1.hist(data_train['freq'])
ax1.set_title('histogram of train record with invalid values')
ax1.set_xlabel('percentage of invalid values ')
ax2 = fig.add_subplot(212)
ax2.hist(data_test['freq'])
ax2.set_title('histogram of test records with invalid values')
ax2.set_xlabel('percentage of invalid values ')
fig.savefig('./Proj_2_export/export_graph/data_histogram_records_invalidvalues.png')


data_train = data_train[data_train['freq']<.3].drop(['freq'], axis = True).reset_index(drop= True)
data_train.to_csv('./Proj_2_export/export_csv/data_train.csv' , index = False)

data_test= data_test[data_test['freq']<.3].drop(['freq'] , axis = 1).reset_index(drop = True)
data_test.to_csv('./Proj_2_export/export_csv/data_test.csv' , index = False)

"""
#Imputation 
from sklearn.impute import SimpleImputer

from fancyimpute import KNN

data_train_knn = pd.DataFrame(KNN(k=6).fit_transform(np.array(data_train.iloc[:,1:])) , columns = data_train.columns[1:])
data_train_knn['id'] = data_train['id']
cols = data_train_knn.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_train_knn = data_train_knn[cols] 
data_train_knn.to_csv('./Proj_2_export/export_csv/data_train_knn.csv')

data_test_knn = pd.DataFrame(KNN(k=6).fit_transform(np.array(data_test.iloc[:,1:])) , columns = data_test.columns[1:])
data_test_knn['id'] = data_test['id']
cols = data_test_knn.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_test_knn = data_test_knn[cols] 
data_test_knn.to_csv('./Proj_2_export/export_csv/data_test_knn.csv')
"""

# !!! The imputation could be done using KNN from fancyimpute or iterativeimputer , it should be submitted on the grid , my computer crashes
data_train_imp = pd.DataFrame(SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data_train.iloc[:,1:]) ,
                                            columns = data_train.columns[1:]) 
data_train_imp['id'] = data_train['id']
cols = data_train_imp.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_train_imp = data_train_imp[cols] 
data_train_imp.to_csv('./Proj_2_export/export_csv/data_train_imp.csv' , index = False )

data_test_imp = pd.DataFrame(SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data_test.iloc[:,1:]) ,
                                            columns = data_test.columns[1:]) 
data_test_imp['id'] = data_test['id']
cols = data_test_imp.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_test_imp = data_test_imp[cols] 
data_test_imp.to_csv('./Proj_2_export/export_csv/data_test_imp.csv' , index = False )


#constructing KPIs afer doing dropping and imputation on acc-level data
kpi_train = KPI_data(data_train_imp , formula = KPI_formula)
kpi_train.to_csv('./Proj_2_export/export_csv/kpi_train.csv' , index = False)

kpi_test = KPI_data(data_test_imp , formula = KPI_formula)
kpi_test.to_csv('./Proj_2_export/export_csv/kpi_test.csv' , index = False) 


# concatenating the datasets (test and train) again in order to have analysi of missing values at KPI level (at index 57022)
kpi_all = pd.concat ([kpi_train , kpi_test ] , ignore_index = True)
kpi_all.to_csv('./Proj_2_export/export_csv/kpi_all.csv' , index = False)



# Analysis of INF , NAN and Zero  at KPI level , Missing values at this level happens due to 0/0 or number/0 leading to np.nan and np.inf  
    
import matplotlib.pyplot as plt

invalid_values  = pd.DataFrame(kpi_all.isna().sum().reset_index(name ='nan'))
invalid_values['nan'] = invalid_values['nan']/kpi_all.shape[0]
invalid_values['zero'] = np.array((kpi_all == 0).sum())/kpi_all.shape[0]
invalid_values['inf'] = np.array((kpi_all ==np.inf).sum()/kpi_all.shape[0])



fig = plt.figure(figsize=(20,10))
p1 = plt.bar(range(invalid_values.shape[0]) , invalid_values['nan'] , color = 'r',  width = 1 )
p2 = plt.bar(range(invalid_values.shape[0]) , invalid_values['inf'] , bottom =invalid_values['nan'],   color = 'b' ,  width = 1 )
p3 = plt.bar(range(invalid_values.shape[0]) , invalid_values['zero'] , bottom =invalid_values['nan']+ invalid_values['inf'],
        color = 'g' , width =1 , linewidth = .1)

plt.title('KPI invalid values' , loc = 'center')
plt.xlabel('KPI_index')
plt.ylabel('Percentage')
plt.legend((p1[0],p2[0] ,p3[0]) , ('nan' , 'inf' , 'zero'))
fig.savefig('./Proj_2_export/export_graph/kpi_all_invalidvalues.png') 


# Dropping columns with more than 30 percent nan values , from 2019 to 87 ; graph does not show any huge portion of inf or zero 
kpi_all_drop = kpi_all.dropna(axis = 1 , thresh = .7 * kpi_all.shape[0] )
kpi_all_drop.to_csv('./Proj_2_export/export_csv/kpi_all_drop.csv' , index = False )

# kpi_all_drop.to_csv('./Proj_2_export/export_csv/kpi_all_invalidvalues.png' , index = False)
invalid_values_afterdrop  = pd.DataFrame(kpi_all_drop.isna().sum().reset_index(name ='nan'))
invalid_values_afterdrop['nan'] = invalid_values_afterdrop['nan']/kpi_all_drop.shape[0]
invalid_values_afterdrop['zero'] = np.array((kpi_all_drop == 0).sum())/kpi_all_drop.shape[0]
invalid_values_afterdrop['inf'] = np.array((kpi_all_drop ==np.inf).sum()/kpi_all_drop.shape[0])

fig = plt.figure(figsize=(12,8))
p1 = plt.bar(range(invalid_values_afterdrop.shape[0]) , invalid_values_afterdrop['nan'] , color = 'r',  width = 1 )
p2 = plt.bar(range(invalid_values_afterdrop.shape[0]) , invalid_values_afterdrop['inf'] ,
             bottom =invalid_values_afterdrop['nan'],   color = 'b' ,  width = 1 )
p3 = plt.bar(range(invalid_values_afterdrop.shape[0]) , invalid_values_afterdrop['zero'] ,
             bottom =invalid_values_afterdrop['nan']+ invalid_values_afterdrop['inf'],color = 'g' , width =1 , linewidth = .1)

plt.title('KPI invalid values after drop down columns with more than 30 nan' , loc = 'center')
plt.xlabel('KPI_index')
plt.ylabel('Percentage')
plt.legend((p1[0],p2[0] ,p3[0]) , ('nan' , 'inf' , 'zero'))
fig.savefig('./Proj_2_export/export_graph/kpi_all_invalidvalues_afterdrop.png') 

# analysis of invalid values per row

kpi_all_drop['freq'] = (kpi_all_drop.isna().sum(axis=1) + (kpi_all_drop ==np.inf).sum(axis =1 )+
            (kpi_all_drop == 0).sum(axis =1 ))/ kpi_all_drop.shape[1]



kpi_train_drop = kpi_all_drop.iloc[:57022,:]
kpi_test_drop = kpi_all_drop.iloc[57022:,:]


fig = plt.figure(figsize= (12,8))
ax1 = fig.add_subplot(311)
ax1.hist(kpi_all_drop['freq'], color = 'r')
ax1.set_title('histogram of kpi record with invalid values')
ax1.set_xlabel('percentage of invalid values ')
ax2 = fig.add_subplot(312)
ax2.hist(kpi_train_drop['freq'] )
ax2.set_title('histogram of kpi TRAIN record with invalid values')
ax2.set_xlabel('percentage of invalid values ')
ax3 = fig.add_subplot(313)
ax3.hist(kpi_test_drop['freq'])
ax3.set_title('histogram of kpi TEST record with invalid values')
ax3.set_xlabel('percentage of invalid values ')
fig.subplots_adjust(hspace = .8)
plt.show()
fig.savefig('./Proj_2_export/export_graph/kpi_histogram_records_invalidvalues.png')


# kpi train record dropped to 55995 from 57022 = 1.8 % 
#kpi test records dropped to 4516 from 4588 = 1.5 % 
kpi_train_imp = kpi_train_drop[kpi_train_drop['freq']<.3].drop(['freq'], axis = True).reset_index(drop= True)
kpi_test_imp= kpi_test_drop[kpi_test_drop['freq']<.3].drop(['freq'] , axis = 1).reset_index(drop = True)


# replacing np.inf and np.nan with zero ()np.nan due to 0/0 and np.inf duet to number / 0 
#!!! should not we impute all the zeros as well. or how can we say zero means no info 
#!!! if that so , how to distinguish zero value and no info  

kpi_train_imp = kpi_train_imp.replace([np.nan , np.inf , -np.inf] , 0 )    
kpi_test_imp = kpi_test_imp.replace([np.nan , np.inf , -np.inf] , 0 )    

kpi_train_imp.to_csv('./Proj_2_export/export_csv/kpi_train_imp.csv' , index = False)
kpi_test_imp.to_csv('./Proj_2_export/export_csv/kpi_test_imp.csv' , index = False)

assert ((kpi_train_imp == np.inf).sum().sum() == 0 )
assert ((kpi_train_imp == np.nan).sum().sum() == 0 )
assert ((kpi_test_imp == np.inf).sum().sum() == 0 )
assert ((kpi_test_imp == np.nan).sum().sum() == 0 )

#################################################################################################
# Scattering plot of GPNV vs SE 
#################################################################################################

kpi_train_imp = pd.read_csv('./Proj_2_export/export_csv/kpi_train_imp.csv' )
kpi_test_imp = pd.read_csv('./Proj_2_export/export_csv/kpi_test_imp.csv' )
kpi_data_imp = pd.concat([kpi_train_imp , kpi_test_imp ] , ignore_index = True)

data_train_imp = pd.read_csv('./Proj_2_export/export_csv/data_train_imp.csv')
data_test_imp = pd.read_csv('./Proj_2_export/export_csv/data_test_imp.csv')
data_all_imp = pd.concat ([data_train_imp , data_test_imp]  , ignore_index = True)

df = kpi_data_imp.merge(data_all_imp , how = 'left' , on = 'id')

df['dealer'] = df['id'].apply(lambda x: x.split("_")[0]) 
df= df [['id','dealer' ,'Gross Profit Per New Unit - Total CDJR Car and Truck Retail' , 'Sales Effectiveness - CJDR' , 'O27','EXP1' ]]
df['dealer'] = df['dealer'].astype('int64')
dealer_region = pd.read_excel('./dealers.xlsx')
df = df.merge(dealer_region , how = 'left' , left_on='dealer' , right_on='code')

fig = plt.figure(figsize=(12,8))
sns.scatterplot(df['Sales Effectiveness - CJDR'] ,df[ 'Gross Profit Per New Unit - Total CDJR Car and Truck Retail'])
plt.title('GPNV vs SE')
fig.savefig('./Proj_2_export/export_graph/GPNV_SE.png')


#!!! this function is removing the outliers but it shoul correct them 
df = remove_outlier(df)
tresh_O27 = (df['O27'].max()- df['O27'].min() )/3 # O27 : Total CDJR Car and Truck Retail - Sales 
df['Total_Sales_scale'] = df['O27'].apply (lambda x : 'low' if x< tresh_O27 else('med' if (x>=tresh_O27 and x< 2* tresh_O27) else 'high'))
tresh_exp1 = (df['EXP1'].max()- df['EXP1'].min() )/3
df['dealer_size'] = df['EXP1'].apply (lambda x : 'small' if x< tresh_exp1 else('med' if (x>=tresh_exp1 and x< 2* tresh_exp1) else 'big'))

# scatter plot , GPNV vs SE 
# at different level of profit O27 
# different level of size EXP1 
# different level of region 

fig = plt.figure(figsize=(12,8))
sns.scatterplot(df['Sales Effectiveness - CJDR'] ,df[ 'Gross Profit Per New Unit - Total CDJR Car and Truck Retail'])
plt.title('GPNV vs SE')
fig.savefig('./Proj_2_export/export_graph/GPNV_SE_WO_outliers.png')


df_size_gp = df.groupby('dealer_size').count()['dealer'].reset_index(name = 'count')
fig= plt.figure()
plt.pie(df_size_gp['count'] , labels = df_size_gp['dealer_size'], autopct='%1.1f%%')
plt.axis('equal') 
fig.suptitle('Number of Dealers based on their Size')  
fig.savefig('./Proj_2_export/export_graph/piechart_dealersize.png')

fig = plt.figure(figsize=(12,16))
fig.suptitle('GPNV vs SE - Dealer Size Analyisis based on EXP1 account')
gs=GridSpec(2,3)
plt.subplot(gs[0,0:])
sns.scatterplot(df['Sales Effectiveness - CJDR'] ,df[ 'Gross Profit Per New Unit - Total CDJR Car and Truck Retail'],
                hue = df['dealer_size'] , size = df['EXP1'] , marker='.')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.subplot(gs[1,0])
ax1 = plt.scatter(df[df['dealer_size'] == 'small']['Sales Effectiveness - CJDR'] , 
            df[df['dealer_size'] == 'small']['Gross Profit Per New Unit - Total CDJR Car and Truck Retail'] ,
            marker='.')
plt.title('small dealers')
plt.subplot(gs[1,1] )
plt.scatter(df[df['dealer_size'] == 'med']['Sales Effectiveness - CJDR'] , 
            df[df['dealer_size'] == 'med']['Gross Profit Per New Unit - Total CDJR Car and Truck Retail'] , c = 'Orange',
            marker='.' ,)
plt.title('medium dealers')
plt.subplot(gs[1,2])
plt.scatter(df[df['dealer_size'] == 'big']['Sales Effectiveness - CJDR'] , 
            df[df['dealer_size'] == 'big']['Gross Profit Per New Unit - Total CDJR Car and Truck Retail'] , c = 'Green')
plt.title('big dealers')
fig.subplots_adjust(wspace=0.5, hspace=0.2)
fig.savefig('./Proj_2_export/export_graph/GPNV_SE_WO_outliers_dealersize.png')


df_size_gp = df.groupby('Total_Sales_scale').count()['dealer'].reset_index(name = 'count')
fig= plt.figure()
plt.pie(df_size_gp['count'] , labels = df_size_gp['Total_Sales_scale'], autopct='%1.1f%%')
plt.axis('equal')  
fig.suptitle('Number of Dealers based on their Sales and Profitability')   
fig.savefig('./Proj_2_export/export_graph/piechart_dealerSaleScale.png')

fig = plt.figure(figsize=(12,16))
fig.suptitle('GPNV vs SE - Profitability Analyisis based on O27 account')
gs=GridSpec(2,3)
plt.subplot(gs[0,0:])
sns.scatterplot(df['Sales Effectiveness - CJDR'] ,df[ 'Gross Profit Per New Unit - Total CDJR Car and Truck Retail'],
                hue = df['Total_Sales_scale'] , size = df['O27'] , marker='.')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.subplot(gs[1,0])
ax1 = plt.scatter(df[df['Total_Sales_scale'] == 'low']['Sales Effectiveness - CJDR'] , 
            df[df['Total_Sales_scale'] == 'low']['Gross Profit Per New Unit - Total CDJR Car and Truck Retail'] ,
            marker='.')
plt.title('low profitability')
plt.subplot(gs[1,1] )
plt.scatter(df[df['Total_Sales_scale'] == 'med']['Sales Effectiveness - CJDR'] , 
            df[df['Total_Sales_scale'] == 'med']['Gross Profit Per New Unit - Total CDJR Car and Truck Retail'] , c = 'Orange',
            marker='.' ,)
plt.title('medium profitabality')
plt.subplot(gs[1,2])
plt.scatter(df[df['Total_Sales_scale'] == 'high']['Sales Effectiveness - CJDR'] , 
            df[df['Total_Sales_scale'] == 'high']['Gross Profit Per New Unit - Total CDJR Car and Truck Retail'] , c = 'Green')
plt.title('big dealers')
fig.subplots_adjust(wspace=0.5, hspace=0.2)
fig.savefig('./Proj_2_export/export_graph/GPNV_SE_WO_outliers_totalsale.png')

df_region_gp = df.groupby('region').count()['dealer'].reset_index(name = 'count')
fig= plt.figure()
plt.pie(df_region_gp['count'] , labels = df_region_gp['region'], autopct='%1.1f%%')
plt.axis('equal') 
fig.suptitle('Number of Dealers based on their regions')    
fig.savefig('./Proj_2_export/export_graph/piechart_dealer_regions.png')

fig = plt.figure(figsize=(12,16))
fig.suptitle('GPNV vs SE - Profitability Analyisis based on Their regions')
gs=GridSpec(3,3)
k=0
for i in range(3):
    for j in range(3):
        plt.subplot(gs[i,j])
        a = df[df['region'] == df.region.unique()[k] ]
        plt.scatter(a['Sales Effectiveness - CJDR'] , 
        a['Gross Profit Per New Unit - Total CDJR Car and Truck Retail'] ,
        marker='.' )
        plt.title('{} region'.format(df.region.unique()[k]))
        k+=1

       
fig.savefig('./Proj_2_export/export_graph/GPNV_SE_WO_outliers_regional.png')
   
### Groupby summary ????            
    

# Normalizing ; normalizing train set and normalize test set based on train parameters 

kpi_train_norm = kpi_train_imp.copy()
kpi_test_norm = kpi_test_imp.copy()
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(kpi_train_norm.iloc[:,1:])
kpi_train_norm.iloc[:,1:] = scaler.transform(kpi_train_imp.iloc[:,1:])
kpi_test_norm.iloc[:,1:] = scaler.transform(kpi_test_norm.iloc[:,1:])

kpi_train_norm.to_csv('./Proj_2_export/export_csv/kpi_train_norm.csv' , index = False)
kpi_test_norm.to_csv('./Proj_2_export/export_csv/kpi_test_norm.csv' , index = False)



###############################################################################################################
# Learning and training :
#                   At this stage , we add gross profit as a traget value and also sales effectiveness as a dependent value 
#                   to the model. They are derived from KPIs . BUT which gross profit and which sales effectiveness?
#                   ??? should we eliminate the unrelated independent variables ? For instance , If i consider Gros profit of CJDR 
#                       should I removed the account for alfaromeo or anything which is unrelated to this target, same story for 
#                       sales effectiveness
#                   Target Value = 'Gross Profit Per New Unit - Total CDJR Car and Truck Retail'
#                   Contributor = 'Sales Effectiveness - CJDR' 
# 1- Checking the available KPIs at this level to see which Gross Profit and which Sales Effectiveness are available and which 
#   is our choice.  
# 2_ how about the records with target value missing (we should not have any at this stage because all the kpis are imputed and drpped )
# 3- !!! should we do impute the target values considering our dataset is not that much large 
# 4- !!! instead of including the sales effectiveness , we could see th efeature importance of the ones which contribute to kpi saleseffective
###############################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor


kpi_train_norm = pd.read_csv('./Proj_2_export/export_csv/kpi_train_norm.csv' )
kpi_test_norm = pd.read_csv('./Proj_2_export/export_csv/kpi_test_norm.csv' )                                                                        

KPI_formula = pd.read_csv('./Proj_2_export/KPI_formulas.csv') 
kpi_features = pd.DataFrame(kpi_train_norm.columns)
dropped_features = pd.DataFrame(set(KPI_formula['KPI Name']) - set(kpi_features.iloc[:,0]), columns = ['list of dropped kpis'])

# Decide about train and test and target value , should I bring the acc , if that so , they should be normalized !!! 

fig = plt.figure(figsize=(10,20))
ax1 = fig.add_subplot(211)
ax1.scatter(kpi_train_norm['Sales Effectiveness - CJDRCar'],
            kpi_train_norm['Gross Profit Per New Unit - Total CDJR Car and Truck Retail'])
ax1.set_xlabel('sales effectiveness')
ax1.set_ylabel('gross profit ')
ax1.set_title('Scatter plot Sales Effectiveness - CJDRCar VS Gross Profit Per New Unit - Total CDJR Car and Truck Retail ' , loc = 'center')
fig.savefig('./Proj_2_export/export_graph/GP vs SalesEffectiveness.png') 


# !!!detecting outliers 


train_x = kpi_train_norm.drop(['id' , 'Gross Profit Per New Unit - Total CDJR Car and Truck Retail'] , axis=1 )
train_y = kpi_train_norm['Gross Profit Per New Unit - Total CDJR Car and Truck Retail']

test_x = kpi_test_norm.drop(['id' , 'Gross Profit Per New Unit - Total CDJR Car and Truck Retail'] , axis=1 )
test_y = kpi_test_norm['Gross Profit Per New Unit - Total CDJR Car and Truck Retail']


# Random forest 

regr = RandomForestRegressor()
regr.fit(train_x, train_y)

fig = plt.figure(figsize = (12,8))
fig.suptitle('Random Forest ; Gross profit vs prediction' , fontweight = 'bold')
ax1 = fig.add_subplot(211)
ax1.set_title('train')
ax1.scatter(train_y , regr.predict(train_x))
ax1.set_xlabel('actual GP')
ax1.set_ylabel('prediction')
ax1.text ( 20,-10, 'R_square = {}'.format(regr.score(train_x , train_y)) , style = 'italic')
ax2 = fig.add_subplot(212)
ax2.set_title('test')
ax2.scatter(test_y , regr.predict(test_x))
ax2.set_xlabel('actual GP')
ax2.set_ylabel('prediction')
ax2.text ( 9,-2, 'R_square = {}'.format(regr.score(test_x , test_y)) , style = 'italic')

fig.savefig('./Proj_2_export/export_graph/RandomForest_default_GP_vs_prediction.png') 

importance = regr.feature_importances_
indices = np.argsort(regr.feature_importances_)[::-1]
fig= plt.figure(figsize=(12,8))
plt.bar(range(20) , importance[indices[:20]])
plt.xticks(range(20), indices[:20])
plt.text(.4,.7,'fisrt feature : {}'.format(train_x.columns[indices[0]]), style = 'oblique')
plt.text(.3,.6,'second ranked :{}'.format(train_x.columns[indices[1]]) , style = 'oblique')
plt.text(.2,.5,'third ranked : {}'.format(train_x.columns[indices[2]]) , style = 'oblique')
fig.suptitle('Random Forest Feature Importance Analysis' , fontweight = 'bold' )
fig.savefig('./Proj_2_export/export_graph/RandomForest_default_featureanalysis.png')


from sklearn import linear_model 
clf = linear_model.Lasso(alpha = .1)
clf.fit(train_x, train_y)

fig = plt.figure(figsize = (12,8))
fig.suptitle('Lasso Regression ; Gross profit vs prediction' , fontweight = 'bold')
ax1 = fig.add_subplot(211)
ax1.set_title('train')
ax1.scatter(train_y , clf.predict(train_x))
ax1.set_xlabel('actual GP')
ax1.set_ylabel('prediction')
ax1.text ( 20,-10, 'R_square = {}'.format(clf.score(train_x , train_y)) , style = 'italic')
ax2 = fig.add_subplot(212)
ax2.set_title('test')
ax2.scatter(test_y , clf.predict(test_x))
ax2.set_xlabel('actual GP')
ax2.set_ylabel('prediction')
ax2.text ( 9,-2, 'R_square = {}'.format(clf.score(test_x , test_y)) , style = 'italic')
fig.savefig('./Proj_2_export/export_graph/LassoRegression_default_GP_vs_prediction.png') 

coef = clf.coef_
indices = np.argsort(coef)[::-1]
fig= plt.figure(figsize=(12,8))
plt.bar(range(10) , coef[indices[:10]])
plt.xticks(range(10), indices[:10])
plt.text(5,.7,'fisrt feature : {}'.format(train_x.columns[indices[0]]), style = 'oblique')
plt.text(5,.6,'second ranked :{}'.format(train_x.columns[indices[1]]) , style = 'oblique')
plt.text(5,.5,'third ranked : {}'.format(train_x.columns[indices[2]]) , style = 'oblique')
fig.suptitle('Random Forest Feature Importance Analysis' , fontweight = 'bold' )
fig.savefig('./Proj_2_export/export_graph/lasso_default_featureanalysis.png')


#heatmap correlation
corr = train_x.corr()

fig = plt.figure(figsize=(24,24))
sns.heatmap(corr)
fig.savefig('./Proj_2_export/export_graph/correlation_heatmap.png' , bbox_inches='tight' , pad_inches = 2)

# detecting outliers 
# remove high correlated features manually , 
KPIs_to_drop = list(KPI_formula['KPI Name'][:26])
KPIs_to_drop.append(KPI_formula['KPI Name'][27])
KPIs_to_drop +=list(KPI_formula['KPI Name'][201:223])

KPIs_to_keep = list(set(train_x.columns) - set(KPIs_to_drop))

train_x = train_x[KPIs_to_keep]
test_x = test_x[KPIs_to_keep]


#Random Forest 
regr = RandomForestRegressor()
regr.fit(train_x, train_y)

fig = plt.figure(figsize = (12,8))
fig.suptitle('Random Forest ; Gross profit vs prediction' , fontweight = 'bold')
ax1 = fig.add_subplot(211)
ax1.set_title('train')
ax1.scatter(train_y , regr.predict(train_x))
ax1.set_xlabel('actual GP')
ax1.set_ylabel('prediction')
ax1.text ( 20,-10, 'R_square = {}'.format(regr.score(train_x , train_y)) , style = 'italic')
ax2 = fig.add_subplot(212)
ax2.set_title('test')
ax2.scatter(test_y , regr.predict(test_x))
ax2.set_xlabel('actual GP')
ax2.set_ylabel('prediction')
ax2.text ( 9,-2, 'R_square = {}'.format(regr.score(test_x , test_y)) , style = 'italic')
fig.savefig('./Proj_2_export/export_graph/RandomForest_default_GP_vs_prediction_removed_corelation_mannually.png') 

importance = regr.feature_importances_
indices = np.argsort(regr.feature_importances_)[::-1]
fig= plt.figure(figsize=(12,8))
plt.bar(range(20) , importance[indices[:20]])
plt.xticks(range(20), indices[:20])
plt.text(5,.4,'fisrt feature : {}'.format(train_x.columns[indices[0]]), style = 'oblique')
plt.text(5,.3,'second ranked :{}'.format(train_x.columns[indices[1]]) , style = 'oblique')
plt.text(5,.2,'third ranked : {}'.format(train_x.columns[indices[2]]) , style = 'oblique')
fig.suptitle('Random Forest Feature Importance Analysis' , fontweight = 'bold' )
fig.savefig('./Proj_2_export/export_graph/RandomForest_default_Removed_Correlation_mannualy_featureanalysis.png')

# Neural Network 


# Maybe I shoudl not normalize the kpis because acc are already normalized ???



