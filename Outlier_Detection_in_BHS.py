# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:09:12 2019

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
from ast import literal_eval
import plotly
import sys

import warnings
warnings.filterwarnings('ignore')

#Save files in given directory to a list
def list_files(dir):                                                                                                    
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(subdir + "/" + file)                                                                         
    return r 

#r=list_files(r'C:\ms\logs\segments\test')
print(sys.argv[1])
r=list_files(sys.argv[1])
print(r)
#Concatenate all Segment Files
def concatenate_segments(allFiles,path):
    big_data_list=[]
    write_header = True
    for f in allFiles:
        for chunk in pd.read_csv(f,index_col=None, header=0,low_memory=False, chunksize=10 ** 6, quotechar='"',skipinitialspace=True, sep=','):
            chunk.to_csv(path, sep=',', encoding='utf-8', index=False, quoting=csv.QUOTE_ALL, mode="a", header=write_header)
            write_header = False
            del chunk
    del big_data_list
    del allFiles
    
concatenate_segments(r,r'all_segments_test.csv')
print("ok")
    
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

def import_cluster(path):
    return pd.read_csv(path,index_col=None, header=0,low_memory=False, quotechar='"', skipinitialspace=True,sep=',')

result=import_data(r'C:\Users\nlokor\Desktop\submit_vanderlande\all_segments_test.csv')
print("ok2")

mondays=result[result.Weekday == 'Monday'].set_index('Activity')
#print('Mondays are seperated!')
tuesdays=result[result.Weekday == 'Tuesday'].set_index('Activity')
#print('Tuesdays are seperated!')
wednesdays=result[result.Weekday == 'Wednesday'].set_index('Activity')
#print('Wednesdays are seperated!')
thursdays=result[result.Weekday == 'Thursday'].set_index('Activity')
#print('Thursdays are seperated!')
fridays=result[result.Weekday == 'Friday'].set_index('Activity')
#print('Fridays are seperated!')
saturdays=result[result.Weekday == 'Saturday'].set_index('Activity')
#print('Saturdays are seperated!')
sundays=result[result.Weekday == 'Sunday'].set_index('Activity')
#print('Sundays are seperated!')

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

segment_day_score_dict={}
def is_outlier(ys,segment_name,weekday):
    if len(ys)>1:
        median_y = np.median(ys)
        median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
        threshold=0
        if median_absolute_deviation_y == 0:
            modified_z_scores=np.zeros(len(ys))
            
        else:    
            modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                             for y in ys]
    
        day_count=len(set(ys.index.date))
        total=len(ys.index.date)
        try:
            load_thresh=total/day_count
        except:
            load_thresh=0
        
        if load_thresh<=200:
            threshold=160
        else:
            threshold=80
        segment_day_score_dict[(segment_name,weekday)]=np.mean(np.abs(modified_z_scores))
    else:
        modified_z_scores=np.zeros(len(ys))
    return np.abs(modified_z_scores) > threshold


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

#Detection of different outlier patterns - Phase 1
def outlier_type(c):
    if len(c)>0:
        indx=0
        for index, row in c.iterrows():
            try:
                if row.CaseID==c.iloc[0].CaseID:
                    c.set_value(index, 'outlier_type', 1)
                    indx=index
                else:
                    if c[c.index==index].shift(-3, freq='min').index < indx:
                        c.set_value(index, 'outlier_type', 0)
                        indx=index
                    else:
                        c.set_value(index, 'outlier_type', 1)
                        indx=index
            except Exception:
                continue
    return c

def outlier_type_phs2(c):
    if len(c)>0:
        for index, row in c.iterrows():
            loc = c.index.get_loc(index)
            try:
                if len(c)==1:
                    c.set_value(index, 'outlier_type', 2)
                else:
                    if row.CaseID != c.iloc[-1].CaseID and row.CaseID != c.iloc[0].CaseID :
                        if (c.iloc[loc + 1].outlier_type==1 or c.iloc[loc + 1].outlier_type==2 ) and (c.iloc[loc - 1].outlier_type==0 or c.iloc[loc - 1].outlier_type==2) and c.iloc[loc].outlier_type==1 :
                            c.set_value(index, 'outlier_type', 2)
                    if row.CaseID == c.iloc[-1].CaseID:
                        if row.outlier_type==0:
                            continue
                        else:
                            c.set_value(index, 'outlier_type', 2)
                    if row.CaseID == c.iloc[0].CaseID :
                        if c.iloc[loc + 1].outlier_type==1  and c.iloc[loc].outlier_type==1 :
                            c.set_value(index, 'outlier_type', 2)
            except Exception:
                continue
    return c

def fast_bag(c):
    median = np.median(c['Duration'])
    for index, row in c.iterrows():
        loc = c.index.get_loc(index)
        try:
            if (c.iloc[loc].outlier_type==1 or c.iloc[loc].outlier_type==2 or c.iloc[loc].outlier_type==0 ) and c.iloc[loc].Duration < median:
                c.set_value(index, 'outlier_type', 4)
        except Exception:
            continue       
    return c


def normalpoints(c):
    if len(c)>0:
        for index, row in c.iterrows():
            c.set_value(index, 'outlier_type', 3)
    return c


def outlier_type_phs3(c):
    c=c.reset_index()
    for index, row in c.iterrows():
        loc = c.index.get_loc(index)
        try:
            if row.CaseID!=c.iloc[0].CaseID and row.CaseID != c.iloc[-1].CaseID:
                if c.iloc[loc].outlier_type==0 and (c.iloc[loc - 1].outlier_type==3 or c.iloc[loc - 1].outlier_type==4 or c.iloc[loc - 1].outlier_type==2):
                    c.set_value(index, 'outlier_type', 1)
                if c.iloc[loc].outlier_type==1 and (c.iloc[loc + 1].outlier_type==3 or c.iloc[loc + 1].outlier_type==4 or c.iloc[loc + 1].outlier_type==1 or c.iloc[loc + 1].outlier_type==2 ):
                    c.set_value(index, 'outlier_type', 2)
        except Exception:
            continue
    return c


def get_time_diff(c):
    point1=0
    point2=0
    point2_end=0
    firstID=''
    lastID=''
    deltaT=0
   
    for index, row in c.iterrows():
        loc = c.index.get_loc(index)
        poin2_list=[]
        try:
            if row.outlier_type==1:
                point1=index
                firstID=str(row.CaseID)
                for index2,row2 in c[loc+1:].iterrows():
                    if row2.outlier_type==0:
                        point2=index2
                        point2_end=point2+ pd.Timedelta(minutes=row2.Duration)
                        poin2_list.append(point2)
                        lastID=str(row2.CaseID)    
                        continue
                    else:
                        break
            
                PatternID=firstID+lastID
                deltaT=(point2_end-point1) / np.timedelta64(1, 'm')
                c.set_value(index, 'PatternID', PatternID)
                c.set_value(index, 'deltaT', deltaT)
                for e in poin2_list:
                    c.set_value(e, 'PatternID', PatternID)
                    c.set_value(e, 'deltaT', deltaT)
            elif row.outlier_type==2 or row.outlier_type==3 or row.outlier_type==4:
                c.set_value(index, 'deltaT', 0)
                c.set_value(index, 'PatternID', 0)
        except Exception:
            continue
    return c


def blockage (row):
    if row['PatternID']!=0:
        return 1
    else:
        return 0
    
def blockage_median (df):
    median_blockage=np.median(df.deltaT)
    return median_blockage

#Find Outlier Score Distrubution
def dist(df,d):
    Unique_Days = df.date.unique()
    df_days = {elem : pd.DataFrame for elem in Unique_Days}
    for key in df_days.keys():
        df_days[key] = df[:][df.date == key]
        d=is_outlier_dist(df_days[key]['Duration'],d,key)
    return d

def hist(first):
    return np.histogram(first,bins=len(first),density=True)

def f_dist( histogram1 ,histogram2):
    if histogram1!=[0] and histogram2!=[0] :
        return wasserstein_distance(histogram1[0],histogram2[0])
    elif histogram1!=[0] and histogram2==[0] :
        return wasserstein_distance(histogram1[0],np.zeros((1,)))
    elif histogram1==[0] and histogram2!=[0] :
        return wasserstein_distance(histogram2[0],np.zeros((1,)))
    

def compare_clusters(df,key,weekday):
    distance_dict={}
    old_cluster=pd.DataFrame()
    dist={}
    min_cluster=''
    folder=r'dist_clusters'
    key_dict='scores_' + key
    key2 = key.replace(".", "!")
    key2 = key2.replace(":", "!")
    
    old_cluster=import_cluster(Path(folder, str(key2)  + str(weekday)  + '_cluster.csv'))
    old_cluster[['Dist']] = old_cluster[['Dist']].applymap(literal_eval)
    
    dist=is_outlier_dist(df['Duration'],{},key)
    
    for index,row in old_cluster.iterrows():
        distance=f_dist(hist(row.Dist),hist(dist[key_dict][0]))
        distance_dict[row.Cluster_perc]=[distance,row.values[2]]
        
    min_cluster= min(distance_dict, key=distance_dict.get)
    df['cluster']=min_cluster
    df['mean_cluster']=distance_dict[min_cluster][1]
    df['mean_score']=np.mean(dist[key_dict][0])
    df['median_score']=np.median(dist[key_dict][0])
    df['q3']=np.percentile(dist[key_dict][0],75, axis=0)
    df['min_score']=np.min(dist[key_dict][0])
    df['max_score']=np.max(dist[key_dict][0])
   
    
def avg_duration(c):
    if len(c.index)<1:
        return 0
    else:
        return np.mean(c['Duration'])  
        
    

def total_blockage_time(c):
    return np.sum(c['deltaT'])

def total_blockage_bags(c):
    count=0
    for row in c['blockage']:
        if row==1:
            count=count+1
        else:
            continue
    return count

def total_isolated_bags(c):
    count=0
    for row in c['outlier_type']:
        if row==2.0:
            count=count+1
        else:
            continue
    return count

def total_fast_bags(c):
    count=0
    for row in c['outlier_type']:
        if row==4.0:
            count=count+1
        else:
            continue
    return count

def total_bag(c):
    return c['CaseID'].nunique()

def calculate_avg_outlier_segment(ys):
    median_y = np.median(ys)
    
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    if median_absolute_deviation_y == 0:
        modified_z_scores=np.zeros(len(ys))
    else:
        modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
     
    hist_data = [np.abs(modified_z_scores)]
    return np.mean(hist_data)

def number_of_outliers(ys):
    return ys[ys.outlier_type!=3]['CaseID'].count()


def psm_table(c,segment_name,dict_psm):
    dict_psm[segment_name] = {
            "cluster":c['cluster'][0],
            "cluster_mean":c['mean_cluster'][0],
            "min_score":c['min_score'][0],
            "max_score":c['max_score'][0],
            "avg_duration_bags": avg_duration(c),
            "bags_in_blockage": total_blockage_bags(c),
            "total_blockage_time": total_blockage_time(c),
            "avg_blockage_time": avg_duration(c[c['blockage']==1]),
            "isolated_bag_count": total_isolated_bags(c),
            "fast_bag_count": total_fast_bags(c),
            "number_of_outliers": number_of_outliers(c),
            "total_bag": total_bag(c),
            "avg_outlier_score": calculate_avg_outlier_segment(c['Duration']),
            "importance": calculate_avg_outlier_segment(c['Duration'])*number_of_outliers(c),
            "importance_blockage": calculate_avg_outlier_segment(c['Duration'])*avg_duration(c[c['blockage']==1]),
            "date": c['date'][0]
        }
    
def blockage_table(c,segment_name):
    for pattern in c.PatternID.unique():
        pattern_df = c[:][c.PatternID == pattern]
        dict_blockage[segment_name,pattern] = {
                "blockage_duration":pattern_df.deltaT[0],
                "count_of_bags":total_bag(pattern_df),
                "avg_duration":avg_duration(pattern_df),
                "blockage_starting":pattern_df.index[0],
                "blockage_end": pattern_df.index[0] + pd.Timedelta(minutes=pattern_df.deltaT[0]),
                "date":pattern_df.date[0]
            }
			

if len(mondays)>0:			
    all_monday= []
    weekday='Monday'
    df_psm=pd.DataFrame()
    dict_psm={}
    dict_blockage={}
    for key in DataFrameDict_mondays.keys():
        print(key + ' is started')  
        DataFrameDict_mondays[key] = mondays[:][mondays.index == key]
        DataFrameDict_mondays[key]=DataFrameDict_mondays[key].reset_index('Activity')
        DataFrameDict_mondays[key]=DataFrameDict_mondays[key].set_index('Timestamp')
        DataFrameDict_mondays[key] =  DataFrameDict_mondays[key].loc[~ DataFrameDict_mondays[key].index.duplicated(keep='first')]
    
        try:
            DataFrameDict_mondays[key]=outlier_type_phs2(outlier_type(DataFrameDict_mondays[key][is_outlier(DataFrameDict_mondays[key]['Duration'],key,weekday)==True ])).append(normalpoints(outlier_type(DataFrameDict_mondays[key][is_outlier(DataFrameDict_mondays[key]['Duration'],key,weekday)==False ]))).sort_index() 
            print('First phase of outlier pattern discovery is done')
            DataFrameDict_mondays[key]=fast_bag(DataFrameDict_mondays[key])
            print('Fast bag discovery is done')
            DataFrameDict_mondays[key]=DataFrameDict_mondays[key].sort_index()
            DataFrameDict_mondays[key]=outlier_type_phs3(DataFrameDict_mondays[key])
            print('Final phase of outlier discovery is done')
            DataFrameDict_mondays[key]=DataFrameDict_mondays[key].set_index(['Timestamp'])
            DataFrameDict_mondays[key]=get_time_diff(DataFrameDict_mondays[key])
            print('Blockage attribute discovery is done')   
            DataFrameDict_mondays[key]['date']=DataFrameDict_mondays[key].index.date.astype(str)
            DataFrameDict_mondays[key]['blockage']=DataFrameDict_mondays[key].apply(blockage, axis=1)
            DataFrameDict_mondays[key]['median_blockage']=blockage_median(DataFrameDict_mondays[key])
            all_monday.append(DataFrameDict_mondays[key])
            compare_clusters(DataFrameDict_mondays[key],key,weekday)
            print('Cluster defined')
            psm_table(DataFrameDict_mondays[key],key,dict_psm)
    
            blockage_table(DataFrameDict_mondays[key][DataFrameDict_mondays[key]['blockage']==1],key)
            print(key + ' is done')
        except Exception as e: 
            print(e)
    fnl_all_monday= pd.concat(all_monday, axis= 0,sort=True,ignore_index = False)
    fnl_all_monday.to_csv(r'test_day_Monday.csv')
    df_psm=pd.DataFrame.from_dict(dict_psm, orient='index')
    df_psm.index.name='segment_name'
    path=r'analysis_psm_general'
    df_psm.sort_values('avg_outlier_score', ascending=False)
    df_psm.to_csv(os.path.join(path , (str(weekday) + '.csv')))
    if len(dict_blockage)>0:
        df_blockage=pd.DataFrame.from_dict(dict_blockage, orient='index')
        df_blockage=df_blockage.reset_index()
        df_blockage.columns = ['segment_name', 'PatternID','blockage_duration','count_of_bags','avg_duration','blockage_start','blockage_end','date']
        path_blockage=r'blockage_psm_general'

        df_blockage.to_csv(os.path.join(path_blockage , (str(weekday) + '.csv')),index=False)




if len(tuesdays)>0:
    all_tuesday= []
    weekday='Tuesday'
    df_psm=pd.DataFrame()
    dict_psm={}
    dict_blockage={}
    for key in DataFrameDict_tuesdays.keys():
        print(key + ' is started')  
        DataFrameDict_tuesdays[key] = tuesdays[:][tuesdays.index == key]
        DataFrameDict_tuesdays[key]=DataFrameDict_tuesdays[key].reset_index('Activity')
        DataFrameDict_tuesdays[key]=DataFrameDict_tuesdays[key].set_index('Timestamp')
        DataFrameDict_tuesdays[key] =  DataFrameDict_tuesdays[key].loc[~ DataFrameDict_tuesdays[key].index.duplicated(keep='first')]
    
        try:
            DataFrameDict_tuesdays[key]=outlier_type_phs2(outlier_type(DataFrameDict_tuesdays[key][is_outlier(DataFrameDict_tuesdays[key]['Duration'],key,weekday)==True ])).append(normalpoints(outlier_type(DataFrameDict_tuesdays[key][is_outlier(DataFrameDict_tuesdays[key]['Duration'],key,weekday)==False ]))).sort_index() 
            print('First phase of outlier pattern discovery is done')
            DataFrameDict_tuesdays[key]=fast_bag(DataFrameDict_tuesdays[key])
            print('Fast bag discovery is done')
            DataFrameDict_tuesdays[key]=DataFrameDict_tuesdays[key].sort_index()
            DataFrameDict_tuesdays[key]=outlier_type_phs3(DataFrameDict_tuesdays[key])
            print('Final phase of outlier discovery is done')
            DataFrameDict_tuesdays[key]=DataFrameDict_tuesdays[key].set_index(['Timestamp'])
            DataFrameDict_tuesdays[key]=get_time_diff(DataFrameDict_tuesdays[key])
            print('Blockage attribute discovery is done')   
            DataFrameDict_tuesdays[key]['date']=DataFrameDict_tuesdays[key].index.date.astype(str)
            DataFrameDict_tuesdays[key]['blockage']=DataFrameDict_tuesdays[key].apply(blockage, axis=1)
            DataFrameDict_tuesdays[key]['median_blockage']=blockage_median(DataFrameDict_tuesdays[key])
            all_tuesday.append(DataFrameDict_tuesdays[key])
            compare_clusters(DataFrameDict_tuesdays[key],key,weekday)
            print('Cluster defined')
            psm_table(DataFrameDict_tuesdays[key],key,dict_psm)
            
            blockage_table(DataFrameDict_tuesdays[key][DataFrameDict_tuesdays[key]['blockage']==1],key)
            print(key + ' is done')
        except Exception as e: 
            print(e)
            
    fnl_all_tuesday= pd.concat(all_tuesday, axis= 0,sort=True,ignore_index = False)
    fnl_all_tuesday.to_csv(r'test_day_Tuesday.csv')
    df_psm=pd.DataFrame.from_dict(dict_psm, orient='index')
    df_psm.index.name='segment_name'
    path=r'analysis_psm_general'
    df_psm.sort_values('avg_outlier_score', ascending=False)
    df_psm.to_csv(os.path.join(path , (str(weekday) + '.csv')))
    if len(dict_blockage)>0:
        df_blockage=pd.DataFrame.from_dict(dict_blockage, orient='index')
        df_blockage=df_blockage.reset_index()
        df_blockage.columns = ['segment_name', 'PatternID','blockage_duration','count_of_bags','avg_duration','blockage_start','blockage_end','date']
        path_blockage=r'blockage_psm_general'
        
        df_blockage.to_csv(os.path.join(path_blockage , (str(weekday) + '.csv')),index=False)




if len(wednesdays)>0:
    all_wednesday= []
    weekday='Wednesday'
    df_psm=pd.DataFrame()
    dict_psm={}
    dict_blockage={}
    for key in DataFrameDict_wednesdays.keys():
        print(key + ' is started')  
        DataFrameDict_wednesdays[key] = wednesdays[:][wednesdays.index == key]
        DataFrameDict_wednesdays[key]=DataFrameDict_wednesdays[key].reset_index('Activity')
        DataFrameDict_wednesdays[key]=DataFrameDict_wednesdays[key].set_index('Timestamp')
        DataFrameDict_wednesdays[key] =  DataFrameDict_wednesdays[key].loc[~ DataFrameDict_wednesdays[key].index.duplicated(keep='first')]
        
        try:
            DataFrameDict_wednesdays[key]=outlier_type_phs2(outlier_type(DataFrameDict_wednesdays[key][is_outlier(DataFrameDict_wednesdays[key]['Duration'],key,weekday)==True ])).append(normalpoints(outlier_type(DataFrameDict_wednesdays[key][is_outlier(DataFrameDict_wednesdays[key]['Duration'],key,weekday)==False ]))).sort_index() 
            print('First phase of outlier pattern discovery is done')
            DataFrameDict_wednesdays[key]=fast_bag(DataFrameDict_wednesdays[key])
            print('Fast bag discovery is done')
            DataFrameDict_wednesdays[key]=DataFrameDict_wednesdays[key].sort_index()
            DataFrameDict_wednesdays[key]=outlier_type_phs3(DataFrameDict_wednesdays[key])
            print('Final phase of outlier discovery is done')
            DataFrameDict_wednesdays[key]=DataFrameDict_wednesdays[key].set_index(['Timestamp'])
            DataFrameDict_wednesdays[key]=get_time_diff(DataFrameDict_wednesdays[key])
            print('Blockage attribute discovery is done')   
            DataFrameDict_wednesdays[key]['date']=DataFrameDict_wednesdays[key].index.date.astype(str)
            DataFrameDict_wednesdays[key]['blockage']=DataFrameDict_wednesdays[key].apply(blockage, axis=1)
            DataFrameDict_wednesdays[key]['median_blockage']=blockage_median(DataFrameDict_wednesdays[key])
            all_wednesday.append(DataFrameDict_wednesdays[key])
            compare_clusters(DataFrameDict_wednesdays[key],key,weekday)
            print('Cluster defined')
            psm_table(DataFrameDict_wednesdays[key],key,dict_psm)
            
            blockage_table(DataFrameDict_wednesdays[key][DataFrameDict_wednesdays[key]['blockage']==1],key)
            print(key + ' is done')
        except Exception as e: 
            print(e)
    fnl_all_wednesday= pd.concat(all_wednesday, axis= 0,sort=True,ignore_index = False)
    fnl_all_wednesday.to_csv(r'test_day_Wednesday.csv')
    df_psm=pd.DataFrame.from_dict(dict_psm, orient='index')
    df_psm.index.name='segment_name'
    path=r'analysis_psm_general'
    df_psm.sort_values('avg_outlier_score', ascending=False)
    df_psm.to_csv(os.path.join(path , (str(weekday) + '.csv')))
    if len(dict_blockage)>0:
        df_blockage=pd.DataFrame.from_dict(dict_blockage, orient='index')
        df_blockage=df_blockage.reset_index()
        df_blockage.columns = ['segment_name', 'PatternID','blockage_duration','count_of_bags','avg_duration','blockage_start','blockage_end','date']
        path_blockage=r'blockage_psm_general'
        
        df_blockage.to_csv(os.path.join(path_blockage , (str(weekday) + '.csv')),index=False)

if len(thursdays)>0:
    all_thursday= []
    weekday='Thursday'
    df_psm=pd.DataFrame()
    dict_psm={}
    dict_blockage={}
    for key in DataFrameDict_thursdays.keys():
        print(key + ' is started')  
        DataFrameDict_thursdays[key] = thursdays[:][thursdays.index == key]
        DataFrameDict_thursdays[key]=DataFrameDict_thursdays[key].reset_index('Activity')
        DataFrameDict_thursdays[key]=DataFrameDict_thursdays[key].set_index('Timestamp')
        DataFrameDict_thursdays[key] =  DataFrameDict_thursdays[key].loc[~ DataFrameDict_thursdays[key].index.duplicated(keep='first')]
        
        try:
            DataFrameDict_thursdays[key]=outlier_type_phs2(outlier_type(DataFrameDict_thursdays[key][is_outlier(DataFrameDict_thursdays[key]['Duration'],key,weekday)==True ])).append(normalpoints(outlier_type(DataFrameDict_thursdays[key][is_outlier(DataFrameDict_thursdays[key]['Duration'],key,weekday)==False ]))).sort_index() 
            print('First phase of outlier pattern discovery is done')
            DataFrameDict_thursdays[key]=fast_bag(DataFrameDict_thursdays[key])
            print('Fast bag discovery is done')
            DataFrameDict_thursdays[key]=DataFrameDict_thursdays[key].sort_index()
            DataFrameDict_thursdays[key]=outlier_type_phs3(DataFrameDict_thursdays[key])
            print('Final phase of outlier discovery is done')
            DataFrameDict_thursdays[key]=DataFrameDict_thursdays[key].set_index(['Timestamp'])
            DataFrameDict_thursdays[key]=get_time_diff(DataFrameDict_thursdays[key])
            print('Blockage attribute discovery is done')   
            DataFrameDict_thursdays[key]['date']=DataFrameDict_thursdays[key].index.date.astype(str)
            DataFrameDict_thursdays[key]['blockage']=DataFrameDict_thursdays[key].apply(blockage, axis=1)
            DataFrameDict_thursdays[key]['median_blockage']=blockage_median(DataFrameDict_thursdays[key])
            all_thursday.append(DataFrameDict_thursdays[key])
            compare_clusters(DataFrameDict_thursdays[key],key,weekday)
            print('Cluster defined')
            psm_table(DataFrameDict_thursdays[key],key,dict_psm)
            
            blockage_table(DataFrameDict_thursdays[key][DataFrameDict_thursdays[key]['blockage']==1],key)
            print(key + ' is done')
        except Exception as e:
            print(e)
    fnl_all_thursdays= pd.concat(all_thursday, axis= 0,sort=True,ignore_index = False)
    fnl_all_thursdays.to_csv(r'test_day_Thursday.csv')
    df_psm=pd.DataFrame.from_dict(dict_psm, orient='index')
    df_psm.index.name='segment_name'
    path=r'analysis_psm_general'
    df_psm.sort_values('avg_outlier_score', ascending=False)
    df_psm.to_csv(os.path.join(path , (str(weekday) + '.csv')))
    
    if len(dict_blockage)>0:
        df_blockage=pd.DataFrame.from_dict(dict_blockage, orient='index')
        df_blockage=df_blockage.reset_index()
        df_blockage.columns = ['segment_name', 'PatternID','blockage_duration','count_of_bags','avg_duration','blockage_start','blockage_end','date']
        path_blockage=r'blockage_psm_general'
    
        df_blockage.to_csv(os.path.join(path_blockage , (str(weekday) + '.csv')),index=False)
    

if len(fridays)>0:   
    all_friday= []
    weekday='Friday'
    df_psm=pd.DataFrame()
    dict_psm={}
    dict_blockage={}
    for key in DataFrameDict_fridays.keys():
        print(key + ' is started')  
        DataFrameDict_fridays[key] = fridays[:][fridays.index == key]
        DataFrameDict_fridays[key]=DataFrameDict_fridays[key].reset_index('Activity')
        DataFrameDict_fridays[key]=DataFrameDict_fridays[key].set_index('Timestamp')
        DataFrameDict_fridays[key] =  DataFrameDict_fridays[key].loc[~ DataFrameDict_fridays[key].index.duplicated(keep='first')]
        
        try:
            DataFrameDict_fridays[key]=outlier_type_phs2(outlier_type(DataFrameDict_fridays[key][is_outlier(DataFrameDict_fridays[key]['Duration'],key,weekday)==True ])).append(normalpoints(outlier_type(DataFrameDict_fridays[key][is_outlier(DataFrameDict_fridays[key]['Duration'],key,weekday)==False ]))).sort_index() 
            print('First phase of outlier pattern discovery is done')
            DataFrameDict_fridays[key]=fast_bag(DataFrameDict_fridays[key])
            print('Fast bag discovery is done')
            DataFrameDict_fridays[key]=DataFrameDict_fridays[key].sort_index()
            DataFrameDict_fridays[key]=outlier_type_phs3(DataFrameDict_fridays[key])
            print('Final phase of outlier discovery is done')
            DataFrameDict_fridays[key]=DataFrameDict_fridays[key].set_index(['Timestamp'])
            DataFrameDict_fridays[key]=get_time_diff(DataFrameDict_fridays[key])
            print('Blockage attribute discovery is done')   
            DataFrameDict_fridays[key]['date']=DataFrameDict_fridays[key].index.date.astype(str)
            DataFrameDict_fridays[key]['blockage']=DataFrameDict_fridays[key].apply(blockage, axis=1)
            DataFrameDict_fridays[key]['median_blockage']=blockage_median(DataFrameDict_fridays[key])
            all_friday.append(DataFrameDict_fridays[key])
            compare_clusters(DataFrameDict_fridays[key],key,weekday)
            print('Cluster defined')
            psm_table(DataFrameDict_fridays[key],key,dict_psm)
    
            blockage_table(DataFrameDict_fridays[key][DataFrameDict_fridays[key]['blockage']==1],key)
            print(key + ' is done')
        except Exception as e: 
            print(e)
    fnl_all_friday = pd.concat(all_friday, axis= 0,sort=True,ignore_index = False)
    fnl_all_friday.to_csv(r'test_day_Friday.csv')
    df_psm=pd.DataFrame.from_dict(dict_psm, orient='index')
    df_psm.index.name='segment_name'
    path=r'analysis_psm_general'
    df_psm.sort_values('avg_outlier_score', ascending=False)
    df_psm.to_csv(os.path.join(path , (str(weekday) + '.csv')))
    if len(dict_blockage)>0:
        df_blockage=pd.DataFrame.from_dict(dict_blockage, orient='index')
        df_blockage=df_blockage.reset_index()
        df_blockage.columns = ['segment_name', 'PatternID','blockage_duration','count_of_bags','avg_duration','blockage_start','blockage_end','date']
        path_blockage=r'blockage_psm_general'
        
        df_blockage.to_csv(os.path.join(path_blockage , (str(weekday) + '.csv')),index=False)

if len(saturdays)>0:   
    all_saturday= []
    weekday='Saturday'
    df_psm=pd.DataFrame()
    dict_psm={}
    dict_blockage={}
    for key in DataFrameDict_saturdays.keys():
        print(key + ' is started')  
        DataFrameDict_saturdays[key] = saturdays[:][saturdays.index == key]
        DataFrameDict_saturdays[key]=DataFrameDict_saturdays[key].reset_index('Activity')
        DataFrameDict_saturdays[key]=DataFrameDict_saturdays[key].set_index('Timestamp')
        DataFrameDict_saturdays[key] =  DataFrameDict_saturdays[key].loc[~ DataFrameDict_saturdays[key].index.duplicated(keep='first')]
        
        try:
            DataFrameDict_saturdays[key]=outlier_type_phs2(outlier_type(DataFrameDict_saturdays[key][is_outlier(DataFrameDict_saturdays[key]['Duration'],key,weekday)==True ])).append(normalpoints(outlier_type(DataFrameDict_saturdays[key][is_outlier(DataFrameDict_saturdays[key]['Duration'],key,weekday)==False ]))).sort_index() 
            print('First phase of outlier pattern discovery is done')
            DataFrameDict_saturdays[key]=fast_bag(DataFrameDict_saturdays[key])
            print('Fast bag discovery is done')
            DataFrameDict_saturdays[key]=DataFrameDict_saturdays[key].sort_index()
            DataFrameDict_saturdays[key]=outlier_type_phs3(DataFrameDict_saturdays[key])
            print('Final phase of outlier discovery is done')
            DataFrameDict_saturdays[key]=DataFrameDict_saturdays[key].set_index(['Timestamp'])
            DataFrameDict_saturdays[key]=get_time_diff(DataFrameDict_saturdays[key])
            print('Blockage attribute discovery is done')   
            DataFrameDict_saturdays[key]['date']=DataFrameDict_saturdays[key].index.date.astype(str)
            DataFrameDict_saturdays[key]['blockage']=DataFrameDict_saturdays[key].apply(blockage, axis=1)
            DataFrameDict_saturdays[key]['median_blockage']=blockage_median(DataFrameDict_saturdays[key])
            all_saturday.append(DataFrameDict_saturdays[key])
            compare_clusters(DataFrameDict_saturdays[key],key,weekday)
            print('Cluster defined')
            psm_table(DataFrameDict_saturdays[key],key,dict_psm)
    
            blockage_table(DataFrameDict_saturdays[key][DataFrameDict_saturdays[key]['blockage']==1],key)
            print(key + ' is done')
        except Exception as e: 
            print(e)
    fnl_all_saturday = pd.concat(all_saturday, axis= 0,sort=True,ignore_index = False)
    fnl_all_saturday.to_csv(r'test_day_Saturday.csv')
    df_psm=pd.DataFrame.from_dict(dict_psm, orient='index')
    df_psm.index.name='segment_name'
    path=r'analysis_psm_general'
    df_psm.sort_values('avg_outlier_score', ascending=False)
    df_psm.to_csv(os.path.join(path , (str(weekday) + '.csv')))
    if len(dict_blockage)>0:
        df_blockage=pd.DataFrame.from_dict(dict_blockage, orient='index')
        df_blockage=df_blockage.reset_index()
        df_blockage.columns = ['segment_name', 'PatternID','blockage_duration','count_of_bags','avg_duration','blockage_start','blockage_end','date']
        path_blockage=r'blockage_psm_general'
        
        df_blockage.to_csv(os.path.join(path_blockage , (str(weekday) + '.csv')),index=False)




if len(sundays)>0:   
    all_sunday= []
    weekday='Sunday'
    df_psm=pd.DataFrame()
    dict_psm={}
    dict_blockage={}
    for key in DataFrameDict_sundays.keys():
        print(key + ' is started')  
        DataFrameDict_sundays[key] = sundays[:][sundays.index == key]
        DataFrameDict_sundays[key]=DataFrameDict_sundays[key].reset_index('Activity')
        DataFrameDict_sundays[key]=DataFrameDict_sundays[key].set_index('Timestamp')
        DataFrameDict_sundays[key] =  DataFrameDict_sundays[key].loc[~ DataFrameDict_sundays[key].index.duplicated(keep='first')]
        
        try:
            DataFrameDict_sundays[key]=outlier_type_phs2(outlier_type(DataFrameDict_sundays[key][is_outlier(DataFrameDict_sundays[key]['Duration'],key,weekday)==True ])).append(normalpoints(outlier_type(DataFrameDict_sundays[key][is_outlier(DataFrameDict_sundays[key]['Duration'],key,weekday)==False ]))).sort_index() 
            print('First phase of outlier pattern discovery is done')
            DataFrameDict_sundays[key]=fast_bag(DataFrameDict_sundays[key])
            print('Fast bag discovery is done')
            DataFrameDict_sundays[key]=DataFrameDict_sundays[key].sort_index()
            DataFrameDict_sundays[key]=outlier_type_phs3(DataFrameDict_sundays[key])
            print('Final phase of outlier discovery is done')
            DataFrameDict_sundays[key]=DataFrameDict_sundays[key].set_index(['Timestamp'])
            DataFrameDict_sundays[key]=get_time_diff(DataFrameDict_sundays[key])
            print('Blockage attribute discovery is done')   
            DataFrameDict_sundays[key]['date']=DataFrameDict_sundays[key].index.date.astype(str)
            DataFrameDict_sundays[key]['blockage']=DataFrameDict_sundays[key].apply(blockage, axis=1)
            DataFrameDict_sundays[key]['median_blockage']=blockage_median(DataFrameDict_sundays[key])
            all_sunday.append(DataFrameDict_sundays[key])
            compare_clusters(DataFrameDict_sundays[key],key,weekday)
            print('Cluster defined')
            psm_table(DataFrameDict_sundays[key],key,dict_psm)
            
            blockage_table(DataFrameDict_sundays[key][DataFrameDict_sundays[key]['blockage']==1],key)
            print(key + ' is done')
        except Exception as e: 
            print(e)
    fnl_all_sunday = pd.concat(all_sunday, axis= 0,sort=True,ignore_index = False)
    fnl_all_sunday.to_csv(r'test_day_Sunday.csv')
    df_psm=pd.DataFrame.from_dict(dict_psm, orient='index')
    df_psm.index.name='segment_name'
    path=r'analysis_psm_general'
    df_psm.sort_values('avg_outlier_score', ascending=False)
    df_psm.to_csv(os.path.join(path , (str(weekday) + '.csv')))
    if len(dict_blockage)>0:
        df_blockage=pd.DataFrame.from_dict(dict_blockage, orient='index')
        df_blockage=df_blockage.reset_index()
        df_blockage.columns = ['segment_name', 'PatternID','blockage_duration','count_of_bags','avg_duration','blockage_start','blockage_end','date']
        path_blockage=r'blockage_psm_general'
            
        df_blockage.to_csv(os.path.join(path_blockage , (str(weekday) + '.csv')),index=False)
    

