# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:15:31 2019

@author: nlokor
"""

# Define all libraries required for this notebook here
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime as dt
import re
import numpy as np
from numpy import nan
import itertools
import collections
from pathlib import Path
from scipy.stats import wasserstein_distance
import scipy.cluster.hierarchy as hcl
from scipy.spatial.distance import squareform 
from scipy import cluster
import operator
import scipy.stats as stats
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering


import warnings


#def import_data(path):
#    return pd.read_csv(path,index_col=None, header=0,low_memory=False, quotechar='"', skipinitialspace=True,sep=',')
one_file = []
result=pd.DataFrame()
def import_data(path):
    for chunk in pd.read_csv(path,index_col=None, header=0,low_memory=False, chunksize=10 ** 6, quotechar='"', skipinitialspace=True,sep=','):  
        chunk['Timestamp']=pd.to_datetime(chunk['Timestamp'], format="%d-%m-%Y %H:%M:%S.%f")
        #Create a field to keep week of the day information
        chunk['Weekday'] = chunk['Timestamp'].dt.weekday_name  
        chunk['Duration'] = chunk['Duration'].apply(lambda x: (x/(1000*60))%60)
        one_file.append(chunk)
    return pd.concat(one_file, axis= 0,sort=False,ignore_index = True)

#result=import_data(r'all_outlier_friday.csv')
result=import_data(r'All_Segments.csv') 

#result=result.set_index('Activity')

#result['Timestamp']=pd.to_datetime(result['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f")

mondays=result[result.Weekday == 'Monday'].set_index('Activity')
print('Mondays are seperated!')
tuesdays=result[result.Weekday == 'Tuesday'].set_index('Activity')
print('Tuesdays are seperated!')
wednesdays=result[result.Weekday == 'Wednesday'].set_index('Activity')
print('Wednesdays are seperated!')
thursdays=result[result.Weekday == 'Thursday'].set_index('Activity')
print('Thursdays are seperated!')
fridays=result[result.Weekday == 'Friday'].set_index('Activity')
print('Fridays are seperated!')
saturdays=result[result.Weekday == 'Saturday'].set_index('Activity')
print('Saturdays are seperated!')
sundays=result[result.Weekday == 'Sunday'].set_index('Activity')
print('Sundays are seperated!')

UniqueNames_mondays = mondays.index.unique()
UniqueNames_tuesdays = tuesdays.index.unique()
UniqueNames_wednesdays = wednesdays.index.unique()
UniqueNames_thursdays = thursdays.index.unique()
UniqueNames_fridays = fridays.index.unique()
UniqueNames_saturdays = saturdays.index.unique()
UniqueNames_sundays = sundays.index.unique()

DataFrameDict_mondays = {elem : pd.DataFrame for elem in UniqueNames_mondays}
DataFrameDict_tuesdays = {elem : pd.DataFrame for elem in UniqueNames_tuesdays}
DataFrameDict_wednesdays = {elem : pd.DataFrame for elem in UniqueNames_wednesdays}
DataFrameDict_thursdays = {elem : pd.DataFrame for elem in UniqueNames_thursdays}
DataFrameDict_fridays = {elem : pd.DataFrame for elem in UniqueNames_fridays}
DataFrameDict_saturdays = {elem : pd.DataFrame for elem in UniqueNames_saturdays}
DataFrameDict_sundays = {elem : pd.DataFrame for elem in UniqueNames_sundays}


def is_outlier_dist(ys,d,date):
    if len(ys)>1:
        median_y = np.median(ys)
        median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
        if median_absolute_deviation_y == 0:
            modified_z_scores=np.zeros(len(ys))
        else:
            modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                                 for y in ys]
    else:
        modified_z_scores=np.zeros(len(ys))
    hist_data = [np.abs(modified_z_scores)]
    d["scores_{0}".format(date)]=hist_data
    return d

def hist(first):
    return np.histogram(first,bins=len(first))

def f_dist( histogram1 ,histogram2):
    if histogram1!=[0] and histogram2!=[0] :
        return wasserstein_distance(histogram1[0],histogram2[0])
    elif histogram1!=[0] and histogram2==[0] :
        return wasserstein_distance(histogram1[0],np.zeros((1,)))
    elif histogram1==[0] and histogram2!=[0] :
        return wasserstein_distance(histogram2[0],np.zeros((1,)))
    
#Find Outlier Score Distrubution
def dist(df,d):
    Unique_Days = df.date.unique()
    df_days = {elem : pd.DataFrame for elem in Unique_Days}
    for key in df_days.keys():
        df_days[key] = df[:][df.date == key]
        d=is_outlier_dist(df_days[key]['Duration'],d,key)
    return d
        
def create_silhouette_dict(range_n_clusters,X,silhouette_avg_dict):
    for n_cluster in range_n_clusters:
        try:
            clusterer = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='ward')
            cluster_labels = clusterer.fit_predict(X)
            silhouette_avg = silhouette_score(X,cluster_labels)
            silhouette_avg_dict[n_cluster]=silhouette_score(X,cluster_labels)
        except Exception:
            continue
        
def compare_cluster_numbers(silhouette_avg_dict):
    n_cluster=max(silhouette_avg_dict.items(), key=operator.itemgetter(1))[0]
    clusterer_agg = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='ward')
    cluster_labels = clusterer_agg.fit_predict(X)
    d_df=pd.DataFrame()
    d_df= pd.DataFrame.from_dict(d, orient='index')
    d_df=d_df.reset_index()
    d_df['cluster']=cluster_labels
    d_df.columns = ['Day', 'Dist','Cluster']
    d_df=d_df.set_index('Cluster')
    d_df_clusters=d_df.groupby(d_df.index)['Dist'].apply(np.hstack).to_frame().reset_index()
    for i in d_df_clusters.Cluster.unique():
        d_df_clusters.set_value(i, 'mean', np.mean(d_df_clusters[d_df_clusters.Cluster==i]['Dist'].values[0]))
        d_df_clusters.set_value(i, 'median', np.median(d_df_clusters[d_df_clusters.Cluster==i]['Dist'].values[0]))
        d_df_clusters.set_value(i, 'std', np.std(d_df_clusters[d_df_clusters.Cluster==i]['Dist'].values[0]))
        d_df_clusters.set_value(i, 'min', np.min(d_df_clusters[d_df_clusters.Cluster==i]['Dist'].values[0])) 
        d_df_clusters.set_value(i, 'max', np.max(d_df_clusters[d_df_clusters.Cluster==i]['Dist'].values[0]))
        d_df_clusters.set_value(i, 'bag_count', len(d_df_clusters[d_df_clusters.Cluster==i]['Dist'].values[0]))
        d_df_clusters.set_value(i, 'day_count', len(d_df[d_df.index==i]['Day'].unique()))
        #d_df_clusters.set_value(i, 'avg_silhouette_score', avg_silhouette_score(distance_matrix,cluster_labels))
    d_df_clusters["Cluster_Rank"] = d_df_clusters["mean"].rank() 
    d_df_clusters["Cluster_perc"] = round(d_df_clusters["mean"].rank()/ n_cluster,2)
    d_df_clusters['Dist'] = d_df_clusters['Dist'].apply(lambda x: list(x))
    return d_df_clusters

def create_clustering(d,key2,weekday,path): 
    #path=r'dist_clusters'
    key2 = key.replace(".", "!")
    key2 = key2.replace(":", "!")
    column=[]
    for d_key in d.keys():
        column.append(d_key)
        
    column.append('scores')
    column=collections.OrderedDict.fromkeys(column)
    distance_matrix=pd.DataFrame(columns=column)
    distance_matrix['scores']=column
    distance_matrix=distance_matrix.set_index('scores')
    distance_matrix.drop('scores',inplace=True)
    
    for index in distance_matrix.index:
        for column2 in distance_matrix.columns:
            distance=f_dist(hist(d[str(index)][0]),hist(d[str(column2)][0]) )
            distance_matrix[column2][index] = distance
            
    range_n_clusters = [3,4,5,6,7,8,9,10]
    X=distance_matrix
    silhouette_avg_dict={}
    create_silhouette_dict(range_n_clusters,distance_matrix,silhouette_avg_dict)
    if distance_matrix.shape[0]>1:
        
        distance_matrix=pd.DataFrame()
        column=[]
        n_cluster=max(silhouette_avg_dict.items(), key=operator.itemgetter(1))[0]
        clusterer_agg = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='ward')
        cluster_labels = clusterer_agg.fit_predict(X)
        d_df=pd.DataFrame()
        d_df= pd.DataFrame.from_dict(d, orient='index')
        d_df=d_df.reset_index()
        d_df['cluster']=cluster_labels
        d_df.columns = ['Day', 'Dist','Cluster']
        d_df=d_df.set_index('Cluster')
        d_df_clusters=d_df.groupby(d_df.index)['Dist'].apply(np.hstack).to_frame().reset_index()
        for i in d_df_clusters.Cluster.unique():
            d_df_clusters.set_value(i, 'mean', np.mean(d_df_clusters[d_df_clusters.Cluster==i]['Dist'].values[0]))
            d_df_clusters.set_value(i, 'median', np.median(d_df_clusters[d_df_clusters.Cluster==i]['Dist'].values[0]))
            d_df_clusters.set_value(i, 'std', np.std(d_df_clusters[d_df_clusters.Cluster==i]['Dist'].values[0]))
            d_df_clusters.set_value(i, 'min', np.min(d_df_clusters[d_df_clusters.Cluster==i]['Dist'].values[0])) 
            d_df_clusters.set_value(i, 'max', np.max(d_df_clusters[d_df_clusters.Cluster==i]['Dist'].values[0]))
            d_df_clusters.set_value(i, 'bag_count', len(d_df_clusters[d_df_clusters.Cluster==i]['Dist'].values[0]))
            d_df_clusters.set_value(i, 'day_count', len(d_df[d_df.index==i]['Day'].unique()))
            #d_df_clusters.set_value(i, 'avg_silhouette_score', avg_silhouette_score(distance_matrix,cluster_labels))
        d_df_clusters["Cluster_Rank"] = d_df_clusters["mean"].rank() 
        d_df_clusters["Cluster_perc"] = round(d_df_clusters["mean"].rank()/ n_cluster,2)
        d_df_clusters['Dist'] = d_df_clusters['Dist'].apply(lambda x: list(x))
    d_df_clusters.to_csv(Path(path, str(key2)  + str(weekday)  + '_cluster.csv'), index=False)
    

weekday='Monday'
path=r'dist_clusters'
d={}

for key in DataFrameDict_mondays.keys():
    d={}
    print(key + ' is started')   
    DataFrameDict_mondays[key] = mondays[:][mondays.index == key]
    DataFrameDict_mondays[key]=DataFrameDict_mondays[key].reset_index('Activity')
    DataFrameDict_mondays[key]=DataFrameDict_mondays[key].set_index('Timestamp')
    DataFrameDict_mondays[key]['date']=DataFrameDict_mondays[key].index.date.astype(str)
    d=dist(DataFrameDict_mondays[key],{})
    try:
        create_clustering(d,key,weekday,path)
        print(key + ' is done')
    except Exception:
        continue
		
weekday='Tuesday'
path=r'C:\Users\nlokor\Desktop\submit_vanderlande\dist_clusters'
d={}		
for key in DataFrameDict_tuesdays.keys():
    d={}
    print(key + ' is started')   
    DataFrameDict_tuesdays[key] = tuesdays[:][tuesdays.index == key]
    DataFrameDict_tuesdays[key]=DataFrameDict_tuesdays[key].reset_index('Activity')
    DataFrameDict_tuesdays[key]=DataFrameDict_tuesdays[key].set_index('Timestamp')
    DataFrameDict_tuesdays[key]['date']=DataFrameDict_tuesdays[key].index.date.astype(str)
    d=dist(DataFrameDict_tuesdays[key],{})
    try:
        create_clustering(d,key,weekday,path)
        print(key + ' is done')
    except Exception:
        continue
		
weekday='Wednesday'
path=r'C:\Users\nlokor\Desktop\submit_vanderlande\dist_clusters'
d={}		
for key in DataFrameDict_wednesdays.keys():
    d={}
    print(key + ' is started')   
    DataFrameDict_wednesdays[key] = wednesdays[:][wednesdays.index == key]
    DataFrameDict_wednesdays[key]=DataFrameDict_wednesdays[key].reset_index('Activity')
    DataFrameDict_wednesdays[key]=DataFrameDict_wednesdays[key].set_index('Timestamp')
    DataFrameDict_wednesdays[key]['date']=DataFrameDict_wednesdays[key].index.date.astype(str)
    d=dist(DataFrameDict_wednesdays[key],{})
    try:
        create_clustering(d,key,weekday,path)
        print(key + ' is done')
    except Exception:
        continue
		
weekday='Thursday'
path=r'C:\Users\nlokor\Desktop\submit_vanderlande\dist_clusters'
d={}
for key in DataFrameDict_thursdays.keys():
    d={}
    print(key + ' is started')   
    DataFrameDict_thursdays[key] = thursdays[:][thursdays.index == key]
    DataFrameDict_thursdays[key]=DataFrameDict_thursdays[key].reset_index('Activity')
    DataFrameDict_thursdays[key]=DataFrameDict_thursdays[key].set_index('Timestamp')
    DataFrameDict_thursdays[key]['date']=DataFrameDict_thursdays[key].index.date.astype(str)
    d=dist(DataFrameDict_thursdays[key],{})
    try:
        create_clustering(d,key,weekday,path)
        print(key + ' is done')
    except Exception:
        continue
		
#weekday='Friday'
#path=r'dist_clusters'
#d={}
#for key in DataFrameDict_fridays.keys():
#    d={}
#    print(key + ' is started')   
#    DataFrameDict_fridays[key] = result[:][result.index == key]
#    d=dist(DataFrameDict_fridays[key],{})
#    try:
#        create_clustering(d,key,weekday,path)
#        print(key + ' is done')
 #   except Exception:
 #       continue
#		
weekday='Saturday'
path=r'C:\Users\nlokor\Desktop\submit_vanderlande\dist_clusters'
d={}
for key in DataFrameDict_saturdays.keys():
    d={}
    print(key + ' is started')   
    DataFrameDict_saturdays[key] = saturdays[:][saturdays.index == key]
    DataFrameDict_saturdays[key]=DataFrameDict_saturdays[key].reset_index('Activity')
    DataFrameDict_saturdays[key]=DataFrameDict_saturdays[key].set_index('Timestamp')
    DataFrameDict_saturdays[key]['date']=DataFrameDict_saturdays[key].index.date.astype(str)
    d=dist(DataFrameDict_saturdays[key],{})
    try:
        create_clustering(d,key,weekday,path)
        print(key + ' is done')
    except Exception:
        continue   
		
weekday='Sunday'
path=r'C:\Users\nlokor\Desktop\submit_vanderlande\dist_clusters'
d={}
for key in DataFrameDict_sundays.keys():
    d={}
    print(key + ' is started')   
    DataFrameDict_sundays[key] = sundays[:][sundays.index == key]
    DataFrameDict_sundays[key]=DataFrameDict_sundays[key].reset_index('Activity')
    DataFrameDict_sundays[key]=DataFrameDict_sundays[key].set_index('Timestamp')
    DataFrameDict_sundays[key]['date']=DataFrameDict_sundays[key].index.date.astype(str)
    d=dist(DataFrameDict_sundays[key],{})
    try:
        create_clustering(d,key,weekday,path)
        print(key + ' is done')
    except Exception:
        continue   