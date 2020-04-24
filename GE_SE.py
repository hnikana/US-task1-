#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:24:16 2019

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
from sklearn.preprocessing import PolynomialFeatures , StandardScaler
from sklearn import linear_model
import multiprocessing as mp
from sklearn.cluster import KMeans
import time
from sklearn import preprocessing
from mpl_toolkits import mplot3d
import itertools

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


data_dropped= pd.read_csv('./Proj_2_export/export_csv/data_drop.csv')[['id','R27','U27','MV1','EXP1']]

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
#===============================================================================================================    
# if we read whole data  

dataset_acc = pd.read_csv('Proj_2_export/dataset_acc.csv' )
zero_frequency = pd.DataFrame((dataset_acc == 0).sum().reset_index(name = 'frequency'))
zero_frequency['frequency'] = zero_frequency['frequency']/ dataset_acc.shape[0]
missing_freq = dataset_acc.isna().sum().reset_index(name = 'frequency')
missing_freq['frequency'] = missing_freq['frequency']/dataset_acc.shape[0]
zero_missing_freq = pd.merge(missing_freq , zero_frequency , on = 'index' , how = 'left' )
zero_missing_freq ['sum'] = zero_missing_freq['frequency_x']+zero_missing_freq['frequency_y']
columns_to_drop = list(zero_missing_freq[zero_missing_freq['sum']> 0.1]['index'] )
data_dropped = dataset_acc.drop(columns = columns_to_drop)

data_dropped['month']=data_dropped['id'].apply(lambda x : int(x.split('_')[2]))
data_dropped['year']=data_dropped['id'].apply(lambda x : int(x.split('_')[1]))
data_dropped['dealer']=data_dropped['id'].apply(lambda x : (x.split('_')[0]))

     
def account_imputing(dealer):
    df = data_dropped[data_dropped['dealer'] == dealer]
    df.sort_values(['year', 'month' ] , inplace = True)
    df.iloc[:,1:51] = KNN(k=5).fit_transform(np.array(df.iloc[:,1:51])) 
    
    low_whisker = df.iloc[:,1:51].quantile(0.01)
    high_whisker = df.iloc[:,1:51].quantile(0.99)
    
    mask_1 = (df.iloc[:,1:51] < low_whisker)
    df.iloc[:,1:51] = np.where(mask_1 , low_whisker , df.iloc[:,1:51])
    
    mask_2 = (df.iloc[:,1:51] > high_whisker)
    df.iloc[:,1:51] = np.where(mask_2 , high_whisker , df.iloc[:,1:51])
    
    return df


data_imputed = pd.DataFrame()
N = len(data_dropped['dealer'].unique())
for i in range(0,N,100) :
    if i+100 < N :
        j = i+100
    else:
        j = N
        
    start = time.time()
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(account_imputing ,[dealer for dealer in data_dropped['dealer'].unique()[i:j]])
    pool.close()

    acc = pd.DataFrame()
    for df in results:
        acc = pd.concat([acc , df] , axis = 0 , ignore_index= True)
    
    data_imputed = pd.concat([data_imputed , acc] , axis = 0 , ignore_index= True)
    del acc
    
    data_imputed.to_csv('./proj_2_export/export_csv/data_constructor_imputed_full.csv' , index = False)
    
        
duration = time.time()-start    


 
#===============================================================================================================
   
def imp(dealer):
    df = data_dropped[data_dropped['dealer'] == dealer]
    df.sort_values(['year', 'month' ] , inplace = True)
    # If a column is all nan , it would drop the feature so , the dimension wont be matched.
    try:
        df.iloc[:,1:]= SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(np.array(df.iloc[:,1:]))
        
    except:
        return dealer
        
    
    low_whisker = df.iloc[:,1:5].quantile(0.01)
    high_whisker = df.iloc[:,1:5].quantile(0.99)
    
    mask_1 = (df.iloc[:,1:5] < low_whisker)
    df.iloc[:,1:5] = np.where(mask_1 , low_whisker , df.iloc[:,1:5])
    
    
    mask_2 = (df.iloc[:,1:5] > high_whisker)
    df.iloc[:,1:5] = np.where(mask_2 , high_whisker , df.iloc[:,1:5])
    
    return df 

start = time.time()
pool = mp.Pool(mp.cpu_count())
results = pool.map(imp ,[dealer for dealer in data_dropped['dealer'].unique()])
pool.close()

data_imputed = pd.DataFrame()
dealer_noinfo=[]
for df in results:
    if isinstance(df,str):
        dealer_noinfo.append(df)
    else:
        data_imputed = pd.concat([data_imputed , df] , axis = 0 , ignore_index= True)

duration = time.time()-start         

data_imputed.to_csv('./proj_2_export/export_csv/data_constructor_imputed.csv' , index = False)

data_imputed = pd.read_csv('./proj_2_export/export_csv/data_constructor_imputed.csv')


data_imputed['GPNV'] = data_imputed['R27']/data_imputed['U27']
data_imputed['SE'] = data_imputed['MV1'] / data_imputed['EXP1']
data_imputed = data_imputed.replace([np.inf , -np.inf] , np.nan)
data_imputed.dropna(inplace = True)
data_imputed['GPNV_scaled'] = StandardScaler().fit_transform(np.array(data_imputed['GPNV']).reshape(-1,1))
data_imputed['SE_scaled'] = StandardScaler().fit_transform(np.array(data_imputed['SE']).reshape(-1,1))

def GP_SE(dealer):
    df = data_imputed[data_imputed['dealer'] == dealer]
    corr = df[['GPNV', 'SE']].corr().iloc[0,1]
    X = np.array(df['GPNV_scaled']).reshape(-1,1)
    X_2 = PolynomialFeatures(degree =2) .fit_transform(X)
    regress_2 = linear_model.LinearRegression(fit_intercept = False)
    regress_2.fit(X_2, df['SE_scaled'])
    X_3 = PolynomialFeatures(degree =3).fit_transform(X)
    regress_3 = linear_model.LinearRegression(fit_intercept = False)
    regress_3.fit(X_3, df['SE_scaled'])
    k=KMeans(1)
    radius_scaled = max(k.fit_transform(df[['GPNV_scaled','SE_scaled']]))
    centroid_norm = k.cluster_centers_
    k=KMeans(1)
    radius = max(k.fit_transform(df[['GPNV','SE']]))
    centroid = k.cluster_centers_
    collector =[dealer ,df.shape[0] , corr , radius_scaled[0] , radius[0] , centroid_norm[0,0] ,centroid_norm[0,1] , centroid[0,0] , centroid[0,1]]
    
    collector.extend(regress_2.coef_)
    collector.append(regress_2.score(X_2 , df['SE_scaled']))
    collector.extend(regress_3.coef_)
    collector.append(regress_3.score(X_3 , df['SE_scaled']))
    
    return collector

start = time.time()
pool = mp.Pool(mp.cpu_count())
results = pool.map(GP_SE ,[dealer for dealer in data_imputed['dealer'].unique()])
pool.close()
duration = time.time()-start 
coef = pd.DataFrame(results , 
                    columns=['dealer' ,'#ofrecords', 'correlation', 'radius_scaled' , 'radius' , 'GP_mean_scaled' , 'SE_mean_scaled', 'GP_mean' , 'SE_mean',
                             'B0' , 'B1' , 'B2', 'R2_deg2' , 'C0', 'C1', 'C2' , 'C3', 'R2_deg3'])


coef.to_csv('./proj_2_export/export_csv/dealer_GP_SE_pattern.csv' , index = False)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# READING THE DATA 
#-------------------------------------------------------------------------------------------------------------------------------------------------------
coef = pd.read_csv('./proj_2_export/export_csv/dealer_GP_SE_pattern.csv')

low_whisker = coef[['GP_mean_scaled' , 'SE_mean_scaled', 'GP_mean','SE_mean' ]].quantile(0.01)
high_whisker = coef[['GP_mean_scaled' , 'SE_mean_scaled', 'GP_mean','SE_mean' ]].quantile(0.99)
mask_1 = (coef.iloc[:,5:9] < low_whisker)
mask_2 = (coef.iloc[:,5:9] > high_whisker)
coef[['GP_scaled_clip' , 'SE_scaled_clip' , 'GP_clip' , 'SE_clip']] = pd.DataFrame(np.where(mask_1 , low_whisker ,coef.iloc[:,5:9]))
coef.iloc[:,18:] = np.where(mask_2 , high_whisker ,coef.iloc[:,18:])

coef['rho'] =  np.sqrt(np.array(np.power((coef['GP_scaled_clip']-coef['GP_scaled_clip'].min()), 2) + np.array(np.power((coef['SE_scaled_clip']-coef['SE_scaled_clip'].min()), 2))))
coef['theta'] = np.arctan2((coef['SE_scaled_clip']-coef['SE_scaled_clip'].min()), (coef['GP_scaled_clip']-coef['GP_scaled_clip'].min()))


#-------------------------------------------------------------------------------------------------------------------------------------------------------
# check homany dealears we lose .it happens because one of the features might be all nan value for that dealer == >  5 dealers !!! it could be found out 
#  comparing the unique delares in data impute and data dropped, 
# but you will get the error as the type of features are not the same . Anyways just 5 dealers we have lost 
#-------------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# plotting SE_mean vs GP_mean not normalized 
#-------------------------------------------------------------------------------------------------------------------------------------------------------

fig , (ax1, ax2)  = plt.subplots(2,1, figsize = (14,12))  
im = ax1.scatter(coef['GP_mean'] , coef['SE_mean'] , c = coef['R2_deg2']  , cmap = discrete_cmap(10, 'CMRmap') , alpha = .3 , label = '') 
ax1.set_xlabel('GP_mean')
ax1.set_ylabel('SE_mean')
ax1.vlines(coef['GP_mean'].quantile(.01) , coef['SE_mean'].min() , coef['SE_mean'].max() , colors = 'r' , linestyles = 'dashed' , label = 'GP 1% quantile')
ax1.vlines(coef['GP_mean'].quantile(.99) , coef['SE_mean'].min() , coef['SE_mean'].max() , colors = 'r' , linestyles = 'dashed' , label = 'GP 99% quantile')
ax1.hlines(coef['SE_mean'].quantile(.01) , coef['GP_mean'].min() , coef['GP_mean'].max() , colors = 'g', linestyles = 'dashed', label = 'SE 1% quantile')
ax1.hlines(coef['SE_mean'].quantile(.99) , coef['GP_mean'].min() , coef['GP_mean'].max() , colors = 'g', linestyles = 'dashed' ,label = 'SE 99% quantile')
ax1.set_title('scattering visualization with 1% & 99% quantile on each axis')
ax1.legend()
ax2.scatter(coef['GP_clip'] , coef['SE_clip'] , c = coef['R2_deg2']  , cmap = discrete_cmap(10, 'CMRmap') , alpha = .3 , label = '') 
ax2.set_xlabel('GP_mean_clip')
ax2.set_ylabel('SE_mean_clip')
ax2.vlines(coef['GP_clip'].quantile(.5) , coef['SE_clip'].min() , coef['SE_clip'].max() , colors = 'r' , linestyles = 'dashed' , label = 'GP 50% quantile')
ax2.vlines(coef['GP_clip'].quantile(.66) , coef['SE_clip'].min() , coef['SE_clip'].max() , colors = 'r' , linestyles = 'dashed' , label = 'GP 66% quantile')
ax2.hlines(coef['SE_clip'].quantile(.5) , coef['GP_clip'].min() , coef['GP_clip'].max() , colors = 'g', linestyles = 'dashed', label = 'SE 50% quantile')
ax2.hlines(coef['SE_clip'].quantile(.66) , coef['GP_clip'].min() , coef['GP_clip'].max() , colors = 'g', linestyles = 'dashed' ,label = 'SE 66% quantile')
ax2.legend()
ax2.set_title('scattering visualization after correcting outliers with 50% & 66% quantile on each axis ')
#plt.axvspan(coef['GP_clip'].quantile(.66), coef['GP_clip'].max(), facecolor='0.2', alpha=0.5)
#plt.axhspan(coef['SE_clip'].quantile(.66), coef['SE_clip'].max(), facecolor='0.2', alpha=0.5)
plt.fill_between(np.arange(coef['GP_clip'].quantile(.66) , coef['GP_clip'].max() , .01), 
                 coef['SE_clip'].quantile(.66), coef['SE_clip'].max(), color='grey', alpha='0.2')
cbar_ax = fig.add_axes([.95, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cbar_ax)

fig.savefig('./Proj_2_export/export_graph/GP_SE/GP_SE_scattering_quantiles.png' , dpi = 300)

'''====================================================================================================='''

fig = plt.figure(figsize = (12,8))
fig.suptitle('R-squared of polynomial deg -2 distribution' , fontweight = 'bold')

ax1  = fig.add_subplot(211)
_ , bins, patches = ax1.hist(coef['R2_deg3'], 40 , edgecolor='white', linewidth=1 , density= True)
for i in range(0,8): 
    patches[i].set_facecolor('b')
for i in range(8, len(patches)):
    patches[i].set_facecolor('r')
ax1.set_xlim(0,.7)

ax1.set_title('Distribution of R^2')    
ax2 = plt.subplot(212 , sharex = ax1 )
ax2.hist(coef['R2_deg2'], 40 , cumulative=True , histtype = 'step' , density = True)
ax2.vlines(.2, 0 , 1)
ax2.set_title('Cumulative')
ax2.set_xlabel('R^2')
fig.savefig('./Proj_2_export/export_graph/GP_SE/R_squared_distribution.png' , dpi = 300) 

'''====================================================================================================='''
fig = plt.figure(figsize = (12,8))
fig.suptitle('correlation  distribution' , fontweight = 'bold')

ax1  = fig.add_subplot(211)
_ , bins, patches = ax1.hist(coef['correlation'], 40 , edgecolor='white', linewidth=1 , density= True)
for i in range(0,4): 
    patches[i].set_facecolor('b')
for i in range(4, 37):
    patches[i].set_facecolor('r')
for i in range(37,len(patches) ): 
    patches[i].set_facecolor('b')
ax1.set_xlim(-1,1)
ax1.set_xlabel('correlation')
ax1.set_ylabel('density')
  

fig.savefig('./Proj_2_export/export_graph/GP_SE/correlation_distribution.png' , dpi = 300) 

'''====================================================================================================='''

df = coef[(coef['R2_deg2'] > .2)]

fig = plt.figure(figsize = (12,8))
plt.title('scattering of data at strong area with R_squared greater than 0.2')
ax = plt.scatter(df['GP_scaled_clip'] , df['SE_scaled_clip'] , cmap = discrete_cmap(2, 'brg') , 
                 c = df['correlation'] , s = 100*df['radius_scaled'] , alpha = .4)
cbar = fig.colorbar(ax)
cbar.set_label('correlation')
fig.savefig('./Proj_2_export/export_graph/GP_SE/GP_SE_scattering_R2_greater_20.png' , dpi = 300)

'''====================================================================================================='''

fig = plt.figure()
ax = plt.axes(projection='3d')
l = ax.scatter3D(df['GP_scaled_clip'] , df['SE_scaled_clip'] ,  df['R2_deg2'] , cmap = discrete_cmap(2, 'brg') ,
                 c = df['correlation'] , s = 10* df['radius_scaled'])
ax.set_xlabel('GP_scaled')
ax.set_ylabel('SE_scaled')
ax.set_zlabel('R-squared')
cbar = fig.colorbar(l)
cbar.set_label('correlation')
ax.legend()
ax.set_title('3d visualization of GP_SE_Rsquared _CORRalation _RADIUS')

'''====================================================================================================='''

coef_strong = coef[ (coef['GP_clip']> coef['GP_clip'].quantile(.66)) & (coef['SE_clip']> coef['SE_clip'].quantile(.66)) ]
# filtering data to the r_2 more than .2

df = coef_strong[coef_strong['R2_deg2'] > .2]

fig = plt.figure(figsize = (12,8))
plt.title('scattering of data at strong area with R_squared greater than 0.2')
ax = plt.scatter(df['GP_scaled_clip'] , df['SE_scaled_clip'] , cmap = discrete_cmap(2, 'brg') ,
                 c = df['correlation'] , s = 100*df['radius_scaled'] , alpha = .4)
fig.colorbar(ax)
fig.savefig('./Proj_2_export/export_graph/GP_SE/GP_SE_strongquantile.png' , dpi = 300)
# low scattering high correlation , 

'''====================================================================================================='''
xy = (coef['GP_scaled_clip'].min() , coef['SE_scaled_clip'].min())
fig , ax = plt.subplots(figsize = (12,8))  
c1 = plt.Circle(xy , coef['rho'].quantile(.66) ,fill = False ,lw = 5 , linestyle = 'dashed' , edgecolor = 'b' )
c2 = plt.Circle(xy , coef['rho'].quantile(.33) ,fill = False ,lw = 5 , linestyle = 'dashed' , edgecolor = 'b' )
ax.add_artist(c1)
ax.add_artist(c2)

x1 = np.arange(coef['GP_scaled_clip'].min() , coef['GP_scaled_clip'].max() , .1)
x2 = np.arange(coef['GP_scaled_clip'].min() , -.5 , .1)
y1 = (1/3)* (x1-coef['GP_scaled_clip'].min()) + coef['SE_scaled_clip'].min()
y2 = (3)* (x2-coef['GP_scaled_clip'].min()) + coef['SE_scaled_clip'].min()

l1 = plt.plot(x1, y1 , lw = 3 , linestyle = '-.' , color = 'k'  , label = '15" ')
l2 = plt.plot(x2, y2 , lw = 3 , linestyle = '-.' , color = 'r' , label = '75"')
scat= plt.scatter(coef['GP_scaled_clip'] , coef['SE_scaled_clip'] , c = coef['R2_deg2'],
                s = 10*coef['radius_scaled'], cmap = discrete_cmap(10, 'CMRmap') , alpha = .3 , label = '') 
plt.legend()
plt.xlabel('GP_normal')
plt.ylabel('SE_normal')
plt.title('acattering within arc 30 and 60 degree arc and 50% and 66% of distance ')
ax.set_aspect(1.0)
cbar_ax = fig.add_axes([.8, 0.1, 0.03, 0.8])
fig.colorbar(scat, cax=cbar_ax , label = 'R_squared')
fig.savefig('./Proj_2_export/export_graph/GP_SE/GP_SE_arcscatering.png' , dpi =300)

'''====================================================================================================='''

df = coef[(coef['rho'] > coef['rho'].quantile(.66)) & 
          (coef['theta']  >= .3) & (coef['theta']  <= 1.25) & (coef['R2_deg2'] > .1)]

fig = plt.figure(figsize = (12,8))
plt.title('scattering of data at strong area with R_squared greater than 0.2')
ax = plt.scatter(df['GP_scaled_clip'] , df['SE_scaled_clip'] , cmap = discrete_cmap(2, 'brg') , 
                 c = df['correlation'] , s = 100*df['radius_scaled'] , alpha = .4)
fig.colorbar(ax , label = 'correlation')
fig.savefig('./Proj_2_export/export_graph/GP_SE/GP_SE_strongarc.png' , dpi = 300)

'''====================================================================================================='''
# 3d visualization of second order polynomial
fig = plt.figure()
ax = plt.axes(projection='3d')
l = ax.scatter3D(df['B0'] , df['B1'] ,  df['B2'] , cmap = discrete_cmap(2, 'brg') , c = df['correlation'] , s = 10* df['radius_scaled'])
ax.set_xlabel('intercept')
ax.set_ylabel('slop')
ax.set_zlabel('2nd coef')
fig.colorbar(l , label = 'correlation')
ax.legend()
ax.set_title('3d visualization of 2nd order polynomial coeficients')


'''====================================================================================================='''
'''====================================================================================================='''
'''====================================================================================================='''
'''=====================================================================================================''' 
'''=====================================================================================================''' 
# USING WHOLE CONSTRUCTOR to REtrieve as many as kpi possible . Data at acount level imputed by knn = 5 with columns 
# less than 10 percent missing value 
'''=====================================================================================================''' 

acc = pd.read_csv('./proj_2_export/export_csv/data_constructor_imputed_full.csv' )

KPI_formula = pd.read_csv('./Proj_2_export/KPI_formulas.csv')        

     
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


KPI = KPI_data(acc , KPI_formula)
missing_freq = KPI.isna().sum().reset_index(name = 'frequency')
missing_freq['frequency'] = missing_freq['frequency']/KPI.shape[0]
columns_to_drop = list(missing_freq[missing_freq['frequency']> 0.1]['index'] )
KPI= KPI.drop(columns = columns_to_drop) 
KPI['dealer_size'] = acc['EXP1']  
KPI[['month' , 'year' , 'dealer']] = acc[['month' , 'year' , 'dealer']]

#columns_to_keep = [
#        'id', 'dealer' , 'month' , 'year' , 'dealer_size' ,
#        'Gross Profit Per New Unit - Total CDJR Car and Truck Retail', 
#        'Average Sales Price - Total CDJR Car and Truck Retail',
#        'Gross Profit Per Total Retail Used Unit', 
#        'Sales Effectiveness - CJDR' , 
#        'Gross % Sales - New Dept' , 
#        'Gross % Sales - Used Dept'
#        ]

#KPI = KPI[columns_to_keep]
KPI = KPI.replace([np.inf , -np.inf] , np.nan) ####??????????? drop dealers if I include more features tahn selected 


KPI.rename(columns={'Gross Profit Per New Unit - Total CDJR Car and Truck Retail' : 'GPNV' , 
                          'Sales Effectiveness - CJDR' : 'SE' , 
                          'Average Sales Price - Total CDJR Car and Truck Retail' : 'ave_price' ,
                          'Gross Profit Per Total Retail Used Unit' : 'GPU' , 
                          'Gross % Sales - New Dept' : 'GSN' , 
                           'Gross % Sales - Used Dept': 'GSU'} , inplace = True)


 
KPI.dropna(inplace = True)
 
for column in KPI.columns[2:33]:
    name = column + '_norm'
    KPI[name] = StandardScaler().fit_transform(np.array(KPI[column]).reshape(-1,1))
    
KPI['day'] = 1
KPI['date'] = pd.to_datetime(KPI[['year', 'month','day']])
    
KPI.to_csv('./proj_2_export/export_csv/KPI_with_DataConstructedImputedFull.csv' , index = False)








def GP_SE_2(dealer):
    df = KPI[KPI['dealer'] == dealer]
    corr = df[['GPNV', 'SE']].corr().iloc[0,1]
    X = np.array(df['GPNV']).reshape(-1,1)
    X_2 = PolynomialFeatures(degree =2) .fit_transform(X)
    regress_2 = linear_model.LinearRegression(fit_intercept = False)
    regress_2.fit(X_2, df['SE_norm'])
    X_3 = PolynomialFeatures(degree =3).fit_transform(X)
    regress_3 = linear_model.LinearRegression(fit_intercept = False)
    regress_3.fit(X_3, df['SE_norm'])
    k=KMeans(1)
    radius_scaled = max(k.fit_transform(df[['GPNV_norm','SE_norm']]))
    k=KMeans(1)
    radius = max(k.fit_transform(df[['GPNV','SE']]))
    collector =[dealer ,df.shape[0] , corr , radius_scaled[0] , radius[0]]
    
    collector.extend(regress_2.coef_)
    collector.append(regress_2.score(X_2 , df['SE_norm']))
    collector.extend(regress_3.coef_)
    collector.append(regress_3.score(X_3 , df['SE_norm']))
    
    collector.extend(list(df.iloc[:,2:35].mean())) # in order to avoide duplicate 'dealer' column
    collector.extend(list(df.iloc[:,36:].mean()))
    
    
    return collector

start = time.time()
pool = mp.Pool(mp.cpu_count())
results = pool.map(GP_SE_2 ,[dealer for dealer in KPI['dealer'].unique()])
pool.close()

columns_name = ['dealer' ,'#ofrecords', 'correlation', 'radius_scaled' , 'radius' ,'B0' , 'B1' , 'B2', 'R2_deg2' , 'C0', 'C1', 'C2' , 'C3', 'R2_deg3' ]
columns_name.extend(KPI.columns[2:35])
columns_name.extend(KPI.columns[36:])
coef_2 = pd.DataFrame(results , columns=columns_name)
duration = time.time()-start 
coef_2  = coef_2[coef_2['#ofrecords'] > 4]
                        
                        
# aggregating data with zipcode and age and demo of dealers 

dealers_info = pd.read_excel('./Dealer Open Dates.xlsx' , sheet_name = None)

dealers_zipcode = pd.read_excel('Copy of Dealer_zip.xlsx' , sheet_name = 'Sheet1')

dealers_demo = pd.DataFrame()
dealers_demo = pd.concat([dealers_demo, dealers_info['Earlist Open Date'][['DEALER' , 'AGE']]] , ignore_index = True)

dealers_demo = pd.merge(dealers_demo,dealers_info['Demo'][['DEALER' , 'CY_POP' , 'CY_HH' ,'CY_AHI_HH' ]],
                        on='DEALER', how='inner')

dealers_demo = pd.merge(dealers_demo,dealers_zipcode, on='DEALER', how='inner')

dealers_demo.rename(columns={'DEALER' : 'dealer'} , inplace = True)

coef= pd.merge(coef_2,dealers_demo,on='dealer', how='left')


### coef with using just the selected KPIs 
coef_2.to_csv('./proj_2_export/export_csv/dealer_coef_KPI_pattern.csv' , index = False)
### coef with using all the KPIs could be recovered 
coef.to_csv('./proj_2_export/export_csv/dealer_coef_All_KPI_demo_pattern.csv' , index = False)
'''=====================================================================================================''' 

coef = pd.read_csv('./proj_2_export/export_csv/dealer_coef_All_KPI_demo_pattern.csv')

coef= coef[ (coef['SE_norm'] > coef['SE_norm'].quantile(.01)) & (coef['SE_norm'] < coef['SE_norm'].quantile(.99))]
coef= coef[ (coef['GPNV_norm'] > coef['GPNV_norm'].quantile(.01)) & (coef['GPNV_norm'] < coef['GPNV_norm'].quantile(.99))]


####                          GP_SE_scattering.png                      #####


fig= plt.figure(figsize = (14,12))  
ax = plt.scatter(coef['GPNV'] , coef['SE'] , c = coef['R2_deg2']  , cmap = discrete_cmap(10, 'CMRmap') , alpha = .3 , label = '') 
plt.xlabel('GPNV')
plt.ylabel('SE')
plt.title('scattering visualization with 1% & 99% quantile on each axis')
#plt.axvspan(coef['GP_clip'].quantile(.66), coef['GP_clip'].max(), facecolor='0.2', alpha=0.5)
#plt.axhspan(coef['SE_clip'].quantile(.66), coef['SE_clip'].max(), facecolor='0.2', alpha=0.5)
plt.fill_between(np.arange(coef['GPNV'].quantile(.66) , coef['GPNV'].max() , .01), 
                 coef['SE'].quantile(.66), coef['SE'].max(), color='grey', alpha='0.05')
cbar_ax = fig.add_axes([.95, 0.1, 0.03, 0.8])
fig.colorbar(ax, cax=cbar_ax)
fig.savefig('./Proj_2_export/export_graph/GP_SE_2/GP_SE_scattering.png' , dpi = 300)

############################################################################################################
####                          R_2_deg_2_distribution.png                      #####

fig = plt.figure(figsize = (12,8))
fig.suptitle('R-squared of polynomial deg -2 distribution' , fontweight = 'bold')

ax1  = fig.add_subplot(211)
_ , bins, patches = ax1.hist(coef['R2_deg2'], 40 , edgecolor='white', linewidth=1 )
for i in range(0,10): 
    patches[i].set_facecolor('b')
for i in range(10, len(patches)):
    patches[i].set_facecolor('r')
ax1.set_xlim(0,.7)

ax1.set_title('Distribution of R^2 _ degree 2')    
ax2 = plt.subplot(212 , sharex = ax1 )
ax2.hist(coef['R2_deg2'], 40 , cumulative=True , histtype = 'step' , density = True)
ax2.vlines(.2, 0 , 1)
ax2.set_title('Cumulative')
ax2.set_xlabel('R^2')
fig.savefig('./Proj_2_export/export_graph/GP_SE_2/R_2_deg_2_distribution.png' , dpi = 300) 

############################################################################################################
####                          R_2_deg_3_distribution.png                      #####   

fig = plt.figure(figsize = (12,8))
fig.suptitle('R-squared of polynomial deg -3 distribution' , fontweight = 'bold')

ax1  = fig.add_subplot(211)
_ , bins, patches = ax1.hist(coef['R2_deg3'], 40 , edgecolor='white', linewidth=1 )
for i in range(0,9): 
    patches[i].set_facecolor('b')
for i in range(9, len(patches)):
    patches[i].set_facecolor('r')


ax1.set_title('Distribution of R^2 _ degree 3')    
ax2 = plt.subplot(212 , sharex = ax1 )
ax2.hist(coef['R2_deg2'], 40 , cumulative=True , histtype = 'step' , density = True)
ax2.vlines(.2, 0 , 1)
ax2.set_title('Cumulative')
ax2.set_xlabel('R^2')
fig.savefig('./Proj_2_export/export_graph/GP_SE_2/R_2_deg_3_distribution.png' , dpi = 300)     

############################################################################################################
####                          correlation_distribution.png                      #####   
plt.rcParams.update({'font.size': 18})

fig = plt.figure(figsize = (12,8))
fig.suptitle('correlation  distribution' , fontweight = 'bold' , y = 1)

ax1  = fig.add_subplot(211)
_ , bins, patches = ax1.hist(coef['correlation'], 40 , edgecolor='white', linewidth=1)
for i in range(0,8): 
    patches[i].set_facecolor('r')
for i in range(8, 33):
    patches[i].set_facecolor('g')
for i in range(33,len(patches) ): 
    patches[i].set_facecolor('r')
ax1.set_xlim(-1,1)
ax1.set_xlabel('correlation')
ax1.set_ylabel('density')
fig.tight_layout()
  

fig.savefig('./Proj_2_export/export_graph/GP_SE_2/correlation_distribution.png' , dpi = 300) 
############################################################################################################
####                          GP_SE_scttering_strong_region_R_.02.png                      #####  

df = coef[(coef['R2_deg2'] > .19) & (coef['GPNV'] > coef['GPNV'] .quantile(.66)) & (coef['R2_deg2'] > coef['R2_deg2'].quantile(.66))]

fig , (ax1,ax2) = plt.subplots(1,2 , figsize = (14,8) , sharey = True)
plt.suptitle('scattering of datapoints in strong region with R^2 >= 0.2' , fontsize = 16 ,  fontweight = 'bold' )

ax = ax1.scatter(df['GPNV'] , df['SE'] , cmap = discrete_cmap(10, 'brg') , 
                 c = df['correlation'] , s = 100*df['radius_scaled'] , alpha = .4)
ax1.set_ylabel('SE', fontsize = 18)
ax1.set_xlabel('GPNV' , fontsize = 18)
ax1.tick_params(labelsize=10, pad=6)

ax2.scatter(df['GPNV'] , df['SE'] , cmap = discrete_cmap(2, 'brg') , 
                 c = df['correlation'] , s = 100*df['radius_scaled'] , alpha = .4)
ax2.set_xlabel('GPNV' , fontsize = 18)
cbar = fig.colorbar(ax)
cbar.set_label('correlation' , fontsize = 18)

plt.tick_params(labelsize=10, pad=6)

fig.savefig('./Proj_2_export/export_graph/GP_SE_2/GP_SE_scttering_strong_region_R_.02.png' , dpi = 300)


############################################################################################################
####                          GP_SE_scttering_R_.02.png                      #####  

df = coef[(coef['R2_deg2'] > .19)]

fig , (ax1,ax2) = plt.subplots(1,2 , figsize = (14,8) , sharey = True)
plt.suptitle('scattering of all data with R_squared >= 0.2', fontsize = 16 ,  fontweight = 'bold')
ax = ax1.scatter(df['GPNV'] , df['SE'] , cmap = discrete_cmap(10, 'brg') , 
                 c = df['correlation'] , s = 100*df['radius_scaled'] , alpha = .4)
ax1.set_ylabel('SE', fontsize = 18)
ax1.set_xlabel('GPNV' , fontsize = 18)
ax1.tick_params(labelsize=10, pad=6)

ax2.scatter(df['GPNV'] , df['SE'] , cmap = discrete_cmap(2, 'brg') , 
                 c = df['correlation'] , s = 100*df['radius_scaled'] , alpha = .4)
ax2.set_xlabel('GPNV' , fontsize = 18)
cbar = fig.colorbar(ax)
cbar.set_label('correlation')
plt.tick_params(labelsize=10, pad=6)
fig.savefig('./Proj_2_export/export_graph/GP_SE_2/GP_SE_scttering_R_.02.png' , dpi = 300)


############################################################################################################
####                          3dvisualization_GP_SE_R_CORR_Rad                      ##### 
fig = plt.figure()
ax = plt.axes(projection='3d')
l = ax.scatter3D(df['GPNV'] , df['SE'] ,  df['R2_deg2'] , cmap = discrete_cmap(2, 'brg') ,
                 c = df['correlation'] , s = 10* df['radius_scaled'])
ax.set_xlabel('GP_scaled')
ax.set_ylabel('SE_scaled') 
ax.set_zlabel('R-squared')
cbar = fig.colorbar(l)
cbar.set_label('correlation')
ax.legend()
ax.set_title('3d visualization of GP_SE_Rsquared _CORRalation _RADIUS') 

############################################################################################################
####                          3dvisualization_2nd order polynomial coef                     ##### 

fig = plt.figure()
ax = plt.axes(projection='3d')
l = ax.scatter3D(df['B0'] , df['B1'] ,  df['B2'] , cmap = discrete_cmap(2, 'brg') , c = df['correlation'] , s = 10* df['radius_scaled'])
ax.set_xlabel('intercept')
ax.set_ylabel('slop')
ax.set_zlabel('2nd coef')
fig.colorbar(l , label = 'correlation')
ax.legend()
ax.set_title('3d visualization of 2nd order polynomial coeficients')


############################################################################################################
####                          Stacked_radius_correlation.png                      #####  


fig = plt.figure(figsize = (8,6))
plt.title('Stacked histogram of Radius _ correlation' , fontsize = 16 ,  fontweight = 'bold')
plt.hist([df[df['correlation'] > 0]['radius_scaled'] , df[df['correlation'] < 0]['radius_scaled']],
         50 ,  stacked=True , histtype= 'barstacked' ,rwidth=0.8 ,  color = ['g','r'] , label = ['+ve corr','-ve corr'])
plt.xlabel('Radius')
plt.ylabel('Density')
plt.xlim(0,6)
plt.legend()
plt.tick_params(labelsize=12, pad=6)
fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Stacked_radius_correlation.png' , dpi = 300)

############################################################################################################
####                         Joint_radius_correlation.png                      #####  


fig = plt.figure(figsize = (8,6))
plt.title('Joint histogram of Radius _ correlation' , fontsize = 16 ,  fontweight = 'bold')
plt.hist([df[df['correlation'] > 0]['radius_scaled'] , df[df['correlation'] < 0]['radius_scaled']],
         50 ,rwidth=0.8 ,  color = ['g','r'] , label = ['+ve corr','-ve corr'])
plt.xlabel('Radius')
plt.ylabel('Density')
plt.xlim(0,6)
plt.legend()
plt.tick_params(labelsize=12, pad=6)
fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Joint_radius_correlation.png' , dpi = 300)

############################################################################################################
####                          Stacked_GPNV_correlation.png                      #####  

fig = plt.figure(figsize = (8,6))
plt.title('Stacked histogram of GPNV _ correlation' , fontsize = 16 ,  fontweight = 'bold')
plt.hist([df[df['correlation'] > 0]['GPNV'] , df[df['correlation'] < 0]['GPNV']],
         50 ,  stacked=True , rwidth=0.8, histtype= 'barstacked' , color = ['g','r'] , label = ['+ve corr','-ve corr'])
plt.xlabel('GPNV' , fontsize = 12)
plt.ylabel('Density' , fontsize = 12)

plt.legend()
plt.tick_params(labelsize=12, pad=6)
fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Stacked_GPNV_correlation.png' , dpi = 300)


############################################################################################################
####                         Joint_GPNV_correlation.png                      #####  


fig = plt.figure(figsize = (8,6))
plt.title('Joint histogram of GPNV _ correlation' , fontsize = 16 ,  fontweight = 'bold')
plt.hist([df[df['correlation'] > 0]['GPNV'] , df[df['correlation'] < 0]['GPNV']],
         50 , rwidth=.85,  color = ['g','r'] , label = ['+ve corr','-ve corr'])
plt.xlabel('GPNV' , fontsize = 12)
plt.ylabel('Density' , fontsize = 12)

plt.legend()
plt.tick_params(labelsize=12, pad=6)
fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Joint_GPNV_correlation.png' , dpi = 300)

############################################################################################################
####                          Stacked_SE_correlation.png                      #####  

fig = plt.figure(figsize = (8,6))
plt.title('Stacked histogram of SE _ correlation' ,  fontsize = 16 ,  fontweight = 'bold')
plt.hist([df[df['correlation'] > 0]['SE'] , df[df['correlation'] < 0]['SE']],
         50 ,  stacked=True , rwidth=0.8, histtype= 'barstacked' , color = ['g','r'] , label = ['+ve corr','-ve corr'])
plt.xlabel('SE' , fontsize = 12)
plt.ylabel('density' , fontsize = 12)

plt.legend()
plt.tick_params(labelsize=12, pad=6)
fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Stacked_SE_correlation.png' , dpi = 300)

############################################################################################################
####                          Joint_SE_correlation.png                      #####  

fig = plt.figure(figsize = (8,6))
plt.title('Joint histogram of SE _ correlation' ,  fontsize = 16 ,  fontweight = 'bold')
plt.hist([df[df['correlation'] > 0]['SE'] , df[df['correlation'] < 0]['SE']],
         50 , color = ['g','r'] , label = ['+ve corr','-ve corr'])
plt.xlabel('SE' , fontsize = 12)
plt.ylabel('density' , fontsize = 12)

plt.legend()
plt.tick_params(labelsize=12, pad=6)
fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Joint_SE_correlation.png' , dpi = 300)


############################################################################################################
####                          Stacked_size_correlation.png                     #####  


t_1 =  coef['dealer_size'].quantile(.5)
t_2 = coef['dealer_size'].quantile(.66)
coef['size'] = pd.cut(coef['dealer_size'] , [0 , t_1, t_2, np.inf] , labels = ['small' , 'medium' , 'big'])

coef ['binary_corr'] = np.where(coef['correlation'] >  0 , 1 , -1)


df = coef[(coef['R2_deg2'] > .19)]

fig , (ax1,ax2) = plt.subplots(2,1 , figsize = (10,8))
plt.suptitle('Stacked histogram of SIZE _ correlation' , fontsize = 16 , y = 1 ,  fontweight = 'bold')

ax1.hist([df[df['correlation'] > 0]['dealer_size'] , df[df['correlation'] < 0]['dealer_size']], 50 ,
          stacked=True , rwidth=0.8 , histtype= 'barstacked' , color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')

ax1.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)
ax1.legend()

ax2.hist([df[df['correlation'] > 0]['size'] , df[df['correlation'] < 0]['size']], 3,
          stacked=True , rwidth=.8 , histtype= 'barstacked' , color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')
ax2.set_xlabel('size' , fontsize = 12 )
ax2.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)
fig.tight_layout()
fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Stacked_size_correlation.png' , dpi = 300)


############################################################################################################
####                          Joint_size_correlation.png                     #####  


fig , (ax1,ax2) = plt.subplots(2,1 , figsize = (10,8))
plt.suptitle('Joint histogram of SIZE _ correlation' , fontsize = 16 , y = 1, fontweight = 'bold')

ax1.hist([df[df['correlation'] > 0]['dealer_size'] , df[df['correlation'] < 0]['dealer_size']], 50 ,
          rwidth=0.8 , color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')

ax1.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)
ax1.legend()

ax2.hist([df[df['correlation'] > 0]['size'] , df[df['correlation'] < 0]['size']],  3,
           rwidth=.8, color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')
ax2.set_xlabel('size' , fontsize = 12 )
ax2.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)

fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Joint_size_correlation.png' , dpi = 300)



############################################################################################################
####                          Discretizing                    #####  

#def discretize(x,bins):
#    #x is a list 
#    # bins is also a sorted list of cutting point
#    for j in range(len(bins)) :
#        if x > bins[j] and x <= bins[j+1]:
#            return 'level_'+str(j+1)

coef['age_dis'] = pd.cut(coef['AGE'] , [0,5,10,25,50,np.inf] , ['[0-5)' , '[5-10)' , '[10_25)' , '[25,50)' ,'[50 plus)'] )                
#coef['age_dis'] = coef['AGE'].apply(lambda x : discretize(x, [0,5,10,25,50,np.inf]) )   # Age discretize
#coef['POP_dis'] = coef['CY_POP'].apply(lambda x : discretize(x, [0,75000 , 200000,np.inf]) ) # population discretize
coef['POP_dis'] = pd.cut(coef['CY_POP'] , [0,75000 , 200000,np.inf] , ['[0-75k)' , '[75-200)' , '[200 plus)' ] )                

#coef['Inc_dis']  = coef['CY_AHI_HH'].apply(lambda x : discretize(x, [0,60000,90000,np.inf]) ) # annula income discretize
coef['Inc_dis'] = pd.cut(coef['CY_AHI_HH'] , [0,60000,90000,np.inf] ,['[0-60k)' , '[60-90)' , '[90 plus)'])                

#coef['HH_dis'] = coef['CY_HH'].apply(lambda x : discretize(x, [0,13000,70000,np.inf]) ) # # of Households discretize
coef['HH_dis'] = pd.cut(coef['CY_HH'] , [0,13000,70000,np.inf] ,['[0-13k)' , '[13-70)' , '[70 plus)'])  

coef.to_csv('./proj_2_export/export_csv/dealer_coef_All_KPI_demo_discretized_pattern.csv' , index = False)

df = coef[(coef['R2_deg2'] > .19)]
############################################################################################################
####                          Joint_age_correlation.png                     #####  



fig , (ax1,ax2) = plt.subplots(2,1 , figsize = (10,8))
plt.suptitle('Joint histogram of Age of Dealership _ correlation' , fontsize = 16 , y = 1, fontweight = 'bold')

ax1.hist([df[df['correlation'] > 0]['AGE'] , df[df['correlation'] < 0]['AGE']], 50 ,
          rwidth=0.8 , color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')

ax1.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)
ax1.legend()

ax2.hist([df[df['correlation'] > 0]['age_dis'] , df[df['correlation'] < 0]['age_dis']],  5,
           rwidth=.8, color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')
ax2.set_xlabel('Age' , fontsize = 12 )
ax2.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)

fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Joint_age_correlation.png' , dpi = 300)



############################################################################################################
####                          Stacked_age_correlation.png                     #####  



fig , (ax1,ax2) = plt.subplots(2,1 , figsize = (10,8))
plt.suptitle('Stacked histogram of Age of Dealership _ correlation' , fontsize = 16 , y = 1, fontweight = 'bold')

ax1.hist([df[df['correlation'] > 0]['AGE'] , df[df['correlation'] < 0]['AGE']], 50 ,
          rwidth=0.8, stacked=True  , histtype= 'barstacked', color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')

ax1.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)
ax1.legend()

ax2.hist([df[df['correlation'] > 0]['age_dis'] , df[df['correlation'] < 0]['age_dis']],  5,
           stacked=True  , histtype= 'barstacked', rwidth=.8, color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')
ax2.set_xlabel('Age' , fontsize = 12 )
ax2.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)

fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Stacked_age_correlation.png' , dpi = 300)

############################################################################################################
####                          Joint_POP_correlation.png                     #####  



fig , (ax1,ax2) = plt.subplots(2,1 , figsize = (10,8))
plt.suptitle('Joint histogram of city population of Dealership _ correlation' , fontsize = 16 , y = 1, fontweight = 'bold')

ax1.hist([df[df['correlation'] > 0]['CY_POP'] , df[df['correlation'] < 0]['CY_POP']], 50 ,
          rwidth=0.8 , color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')

ax1.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)
ax1.legend()

ax2.hist([df[df['correlation'] > 0]['POP_dis'] , df[df['correlation'] < 0]['POP_dis']],  5,
           rwidth=.8, color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')
ax2.set_xlabel('Population' , fontsize = 12 )
ax2.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)

fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Joint_POP_correlation.png' , dpi = 300)



############################################################################################################
####                          Stacked_POP_correlation.png                     #####  



fig , (ax1,ax2) = plt.subplots(2,1 , figsize = (10,8))
plt.suptitle('Stacked histogram of city population of Dealership _ correlation' , fontsize = 16 , y = 1, fontweight = 'bold')

ax1.hist([df[df['correlation'] > 0]['CY_POP'] , df[df['correlation'] < 0]['CY_POP']], 50 ,
          rwidth=0.8, stacked=True  , histtype= 'barstacked', color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')

ax1.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)
ax1.legend()

ax2.hist([df[df['correlation'] > 0]['POP_dis'] , df[df['correlation'] < 0]['POP_dis']], 
           stacked=True  , histtype= 'barstacked', rwidth=.8, color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')
ax2.set_xlabel('population' , fontsize = 12 )
ax2.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)

fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Stacked_POP_correlation.png' , dpi = 300)    

############################################################################################################
####                          Joint_income_correlation.png                     #####  



fig , (ax1,ax2) = plt.subplots(2,1 , figsize = (10,8))
plt.suptitle('Joint histogram of Annual Income of Dealership city _ correlation' , fontsize = 16 , y = 1, fontweight = 'bold')

ax1.hist([df[df['correlation'] > 0]['CY_AHI_HH'] , df[df['correlation'] < 0]['CY_AHI_HH']], 50 ,
          rwidth=0.8 , color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')

ax1.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)
ax1.legend()

ax2.hist([df[df['correlation'] > 0]['Inc_dis'] , df[df['correlation'] < 0]['Inc_dis']],  5,
           rwidth=.8, color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')
ax2.set_xlabel('Income' , fontsize = 12 )
ax2.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)

fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Joint_income_correlation.png' , dpi = 300)



############################################################################################################
####                          Stacked_income_correlation.png                     #####  



fig , (ax1,ax2) = plt.subplots(2,1 , figsize = (10,8))
plt.suptitle('Stacked histogram of Annual Income of Dealership city _ correlation' , fontsize = 16 , y = 1, fontweight = 'bold')

ax1.hist([df[df['correlation'] > 0]['CY_AHI_HH'] , df[df['correlation'] < 0]['CY_AHI_HH']], 50 ,
          rwidth=0.8, stacked=True  , histtype= 'barstacked', color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')

ax1.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)
ax1.legend()

ax2.hist([df[df['correlation'] > 0]['Inc_dis'] , df[df['correlation'] < 0]['Inc_dis']], 
           stacked=True  , histtype= 'barstacked', rwidth=.8, color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')
ax2.set_xlabel('Income' , fontsize = 12 )
ax2.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)

fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Stacked_income_correlation.png' , dpi = 300)  


############################################################################################################
####                          Joint_HouseHold_correlation.png                     #####  



fig , (ax1,ax2) = plt.subplots(2,1 , figsize = (10,8))
plt.suptitle('Joint histogram of number of households of Dealership city _ correlation' , fontsize = 16 , y = 1, fontweight = 'bold')

ax1.hist([df[df['correlation'] > 0]['CY_HH'] , df[df['correlation'] < 0]['CY_HH']], 50 ,
          rwidth=0.8 , color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')

ax1.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)
ax1.legend()

ax2.hist([df[df['correlation'] > 0]['HH_dis'] , df[df['correlation'] < 0]['HH_dis']],  5,
           rwidth=.8, color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')
ax2.set_xlabel('number of house holds' , fontsize = 12 )
ax2.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)

fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Joint_household_correlation.png' , dpi = 300)



############################################################################################################
####                          Stacked_HouseHold_correlation.png                     #####  



fig , (ax1,ax2) = plt.subplots(2,1 , figsize = (10,8))
plt.suptitle('Stacked histogram of number of households of Dealership city _ correlation' , fontsize = 16 , y = 1, fontweight = 'bold')

ax1.hist([df[df['correlation'] > 0]['CY_HH'] , df[df['correlation'] < 0]['CY_HH']], 50 ,
          rwidth=0.8, stacked=True  , histtype= 'barstacked', color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')

ax1.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)
ax1.legend()

ax2.hist([df[df['correlation'] > 0]['HH_dis'] , df[df['correlation'] < 0]['HH_dis']], 
           stacked=True  , histtype= 'barstacked', rwidth=.8, color = ['g','r'] , label = ['+ve corr','-ve corr'] , align = 'mid')
ax2.set_xlabel('number of house holds' , fontsize = 12 )
ax2.set_ylabel('density' , fontsize = 12)
plt.tick_params(labelsize=12, pad=6)

fig.savefig('./Proj_2_export/export_graph/GP_SE_2/Stacked_household_correlation.png' , dpi = 300)      



############################################################################################################
####                          Lasso Model correlation vs fetures all normal                     #####  
dealer_state = pd.read_excel('./dealers.xlsx')

coef= pd.merge(coef ,dealer_state,on='dealer', how='left')

coef.to_csv('./proj_2_export/export_csv/dealer_coef_All_KPI_demo_discretized_pattern.csv' , index = False)


index_1 = df.columns.get_loc('GPNV_norm')
index_2 = df.columns.get_loc('Sales Effectiveness - CJDRTruck_norm')

X = pd.concat([df.iloc[:,index_1:index_2] , pd.get_dummies(df.iloc[:,83:88])  , pd.get_dummies(df.iloc[:,90])] ,
               axis = 1 , sort = False )

y = df['correlation']

reg = linear_model.Lasso(alpha= .01)
reg.fit(X, y)
score = reg.score(X,y)

fig = plt.figure(figsize = (12,8))
fig.suptitle('Lasso Regression ; Gross profit vs prediction' , fontweight = 'bold')
ax1 = fig.add_subplot(211)
ax1.set_title('prediction vs actual')
ax1.set_xlabel = ('actual correlation')
ax1.set_ylabel = ('prediction')
ax1.scatter(df['correlation'] ,reg.predict(X))

ax1.text ( -.9,-.2, 'R_square = {}'.format(score , style = 'italic'))
coefficients = reg.coef_ 
indices = np.argsort(coefficients)[::-1]
ax2 = fig.add_subplot(212)
ax2.bar(range(10) , coefficients[indices[:10]])
ax2.set_xlim(0,10)
plt.xticks(range(7), X.columns[indices[:10]])
fig.suptitle('Lasso Regression Lambda = .01 Feature Importance Analysis' , fontweight = 'bold' )

fig.savefig('./Proj_2_export/export_graph/GP_SE_2/LassoRegression_default_correlation.png') 


############################################################################################################


def GP_SE_train(dealer):
    df = KPI_train[KPI_train['dealer'] == dealer]
    corr = df[['GPNV', 'SE']].corr().iloc[0,1]
    X = np.array(df['GPNV']).reshape(-1,1)
    X_2 = PolynomialFeatures(degree =2) .fit_transform(X)
    regress_2 = linear_model.LinearRegression(fit_intercept = False)
    regress_2.fit(X_2, df['SE_norm'])
    X_3 = PolynomialFeatures(degree =3).fit_transform(X)
    regress_3 = linear_model.LinearRegression(fit_intercept = False)
    regress_3.fit(X_3, df['SE_norm'])
    k=KMeans(1)
    radius_scaled = max(k.fit_transform(df[['GPNV_norm','SE_norm']]))
    k=KMeans(1)
    radius = max(k.fit_transform(df[['GPNV','SE']]))
    collector =[dealer ,df.shape[0] , corr , radius_scaled[0] , radius[0]]
    
    collector.extend(regress_2.coef_)
    collector.append(regress_2.score(X_2 , df['SE_norm']))
    collector.extend(regress_3.coef_)
    collector.append(regress_3.score(X_3 , df['SE_norm']))
    
    collector.extend(list(df.iloc[:,2:35].mean())) # in order to avoide duplicate 'dealer' column
    collector.extend(list(df.iloc[:,36:].mean()))
    
    
    return collector

def GP_SE_test(dealer):
    df = KPI_test[KPI_test['dealer'] == dealer]
    corr = df[['GPNV', 'SE']].corr().iloc[0,1]
    X = np.array(df['GPNV']).reshape(-1,1)
    X_2 = PolynomialFeatures(degree =2) .fit_transform(X)
    regress_2 = linear_model.LinearRegression(fit_intercept = False)
    regress_2.fit(X_2, df['SE_norm'])
    X_3 = PolynomialFeatures(degree =3).fit_transform(X)
    regress_3 = linear_model.LinearRegression(fit_intercept = False)
    regress_3.fit(X_3, df['SE_norm'])
    k=KMeans(1)
    radius_scaled = max(k.fit_transform(df[['GPNV_norm','SE_norm']]))
    k=KMeans(1)
    radius = max(k.fit_transform(df[['GPNV','SE']]))
    collector =[dealer ,df.shape[0] , corr , radius_scaled[0] , radius[0]]
    
    collector.extend(regress_2.coef_)
    collector.append(regress_2.score(X_2 , df['SE_norm']))
    collector.extend(regress_3.coef_)
    collector.append(regress_3.score(X_3 , df['SE_norm']))
    
    collector.extend(list(df.iloc[:,2:35].mean())) # in order to avoide duplicate 'dealer' column
    collector.extend(list(df.iloc[:,36:].mean()))
    
    
    return collector

KPI_test = KPI[KPI['date']>='2018-3-1']
KPI_train = KPI[KPI['date']<'2018-3-1']

# coef_train
start = time.time()
pool = mp.Pool(mp.cpu_count())
results = pool.map(GP_SE_train ,[dealer for dealer in KPI_train['dealer'].unique()])
pool.close()

columns_name = ['dealer' ,'#ofrecords', 'correlation', 'radius_scaled' , 'radius' ,'B0' , 'B1' , 'B2', 'R2_deg2' , 'C0', 'C1', 'C2' , 'C3', 'R2_deg3' ]
columns_name.extend(KPI.columns[2:35])
columns_name.extend(KPI.columns[36:68])
coef_train = pd.DataFrame(results , columns=columns_name)
duration = time.time()-start 
coef_train  = coef_train[coef_train['#ofrecords'] > 4]

#coef_test
start = time.time()
pool = mp.Pool(mp.cpu_count())
results = pool.map(GP_SE_test ,[dealer for dealer in KPI_test['dealer'].unique()])
pool.close()

columns_name = ['dealer' ,'#ofrecords', 'correlation', 'radius_scaled' , 'radius' ,'B0' , 'B1' , 'B2', 'R2_deg2' , 'C0', 'C1', 'C2' , 'C3', 'R2_deg3' ]
columns_name.extend(KPI.columns[2:35])
columns_name.extend(KPI.columns[36:68])
coef_test = pd.DataFrame(results , columns=columns_name)
duration = time.time()-start 
coef_test  = coef_test[coef_test['#ofrecords'] > 4]                                    
                        
                        
# aggregating data with zipcode and age and demo of dealers 

dealers_info = pd.read_excel('./Dealer Open Dates.xlsx' , sheet_name = None)

dealers_zipcode = pd.read_excel('Copy of Dealer_zip.xlsx' , sheet_name = 'Sheet1')

dealers_demo = pd.DataFrame()
dealers_demo = pd.concat([dealers_demo, dealers_info['Earlist Open Date'][['DEALER' , 'AGE']]] , ignore_index = True)

dealers_demo = pd.merge(dealers_demo,dealers_info['Demo'][['DEALER' , 'CY_POP' , 'CY_HH' ,'CY_AHI_HH' ]],
                        on='DEALER', how='inner')

dealers_demo = pd.merge(dealers_demo,dealers_zipcode, on='DEALER', how='inner')

dealers_demo.rename(columns={'DEALER' : 'dealer'} , inplace = True)

coef_train= pd.merge(coef_train,dealers_demo,on='dealer', how='left')
coef_test= pd.merge(coef_test,dealers_demo,on='dealer', how='left')



### coef with using just the selected KPIs 
### coef with using all the KPIs could be recovered 
coef_train.to_csv('./proj_2_export/export_csv/dealer_coef_train_All_KPI_demo_pattern.csv' , index = False)

coef_test.to_csv('./proj_2_export/export_csv/dealer_coef_test_All_KPI_demo_pattern.csv' , index = False)

############################################################################################################

'''
revision: creating datapoints based on 12 month records per dealer 
so each dealer would have 3 datapoints instead of one , in the impute data set we have 39 month per dealer 
data_imputed.groupby('dealer').count()['year'].unique()
'''



KPI = pd.read_csv('./proj_2_export/export_csv/KPI_with_DataConstructedImputedFull.csv' ) 
KPI.sort_values(['year' , 'month' ], inplace = True)

def GP_SE_3(dealer):
    
    collectors = []
    for i in range(3):                   
        df = KPI[KPI['dealer'] == dealer][i*12:(i+1)*12]
        if df.shape[0] != 0 :
            corr = df[['GPNV', 'SE']].corr().iloc[0,1]
            X = np.array(df['GPNV']).reshape(-1,1)
            print (X.shape)
            X_2 = PolynomialFeatures(degree =2) .fit_transform(X)
            regress_2 = linear_model.LinearRegression(fit_intercept = False)
            regress_2.fit(X_2, df['SE_norm'])
            X_3 = PolynomialFeatures(degree =3).fit_transform(X)
            regress_3 = linear_model.LinearRegression(fit_intercept = False)
            regress_3.fit(X_3, df['SE_norm'])
            k=KMeans(1)
            radius_scaled = max(k.fit_transform(df[['GPNV_norm','SE_norm']]))
            k=KMeans(1)
            radius = max(k.fit_transform(df[['GPNV','SE']]))
            collector =[dealer ,df.shape[0] , corr , radius_scaled[0] , radius[0]]
        
            collector.extend(regress_2.coef_)
            collector.append(regress_2.score(X_2 , df['SE_norm']))
            collector.extend(regress_3.coef_)
            collector.append(regress_3.score(X_3 , df['SE_norm']))
    
        
            collector.extend(list(df.iloc[:,2:35].mean())) # in order to avoide duplicate 'dealer' column
            collector.extend(list(df.iloc[:,36:].mean()))
            
        
            collectors.append(collector)
        
    return collectors

start = time.time()
pool = mp.Pool(mp.cpu_count())
results = pool.map(GP_SE_3 ,[dealer for dealer in KPI['dealer'].unique()])
pool.close()
duration = time.time()-start 

merged = list(itertools.chain.from_iterable(results))
columns_name = ['dealer' ,'#ofrecords', 'correlation', 'radius_scaled' , 'radius' ,'B0' , 'B1' , 'B2', 'R2_deg2' , 'C0', 'C1', 'C2' , 'C3', 'R2_deg3' ]
columns_name.extend(KPI.columns[2:35])
columns_name.extend(KPI.columns[36:68])
coef_revised = pd.DataFrame(merged , columns=columns_name)
duration = time.time()-start 
coef_revised  = coef_revised[coef_revised['#ofrecords'] > 4]
                        
                        
# aggregating data with zipcode and age and demo of dealers 

dealers_info = pd.read_excel('./Dealer Open Dates.xlsx' , sheet_name = None)

dealers_zipcode = pd.read_excel('Copy of Dealer_zip.xlsx' , sheet_name = 'Sheet1')

dealers_demo = pd.DataFrame()
dealers_demo = pd.concat([dealers_demo, dealers_info['Earlist Open Date'][['DEALER' , 'AGE']]] , ignore_index = True)

dealers_demo = pd.merge(dealers_demo,dealers_info['Demo'][['DEALER' , 'CY_POP' , 'CY_HH' ,'CY_AHI_HH' ]],
                        on='DEALER', how='inner')

dealers_demo = pd.merge(dealers_demo,dealers_zipcode, on='DEALER', how='inner')

dealers_demo.rename(columns={'DEALER' : 'dealer'} , inplace = True)

coef_revised = pd.merge(coef_revised ,dealers_demo,on='dealer', how='left')


### coef with using just the selected KPIs 
#coef_2.to_csv('./proj_2_export/export_csv/dealer_coef_KPI_pattern.csv' , index = False)
### coef with using all the KPIs could be recovered 
coef_revised['age_dis'] = pd.cut(coef_revised['AGE'] , [0,5,10,25,50,np.inf] , ['[0-5)' , '[5-10)' , '[10_25)' , '[25,50)' ,'[50 plus)'] )                
#coef['age_dis'] = coef['AGE'].apply(lambda x : discretize(x, [0,5,10,25,50,np.inf]) )   # Age discretize
#coef['POP_dis'] = coef['CY_POP'].apply(lambda x : discretize(x, [0,75000 , 200000,np.inf]) ) # population discretize
coef_revised['POP_dis'] = pd.cut(coef_revised['CY_POP'] , [0,75000 , 200000,np.inf] , ['[0-75k)' , '[75-200)' , '[200 plus)' ] )                

#coef['Inc_dis']  = coef['CY_AHI_HH'].apply(lambda x : discretize(x, [0,60000,90000,np.inf]) ) # annula income discretize
coef_revised['Inc_dis'] = pd.cut(coef_revised['CY_AHI_HH'] , [0,60000,90000,np.inf] ,['[0-60k)' , '[60-90)' , '[90 plus)'])                

#coef['HH_dis'] = coef['CY_HH'].apply(lambda x : discretize(x, [0,13000,70000,np.inf]) ) # # of Households discretize
coef_revised['HH_dis'] = pd.cut(coef_revised['CY_HH'] , [0,13000,70000,np.inf] ,['[0-13k)' , '[13-70)' , '[70 plus)'])  

dealer_state = pd.read_excel('./dealers.xlsx')

coef_revised= pd.merge(coef_revised ,dealer_state,on='dealer', how='left')



coef_revised.to_csv('./proj_2_export/export_csv/dealer_coef_All_KPI_demo_discretized_pattern_revised.csv' , index = False)



