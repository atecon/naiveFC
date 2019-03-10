#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Sample script for running the naiveFC package for Python
    
    @author: Artur Tarassow
"""

machine = 1         # 1=AT-home, 2=DSB

# Set working dir
#================
import os
os.getcwd()

if machine==1:
    # ADJUST THIS PATH ACCORDING TO YOUR MACHINE
    os.chdir("/home/at/git/naiveFC/python/src")
    
elif machine==2:
    os.chdir("/home/ninja/BrainLocalData/shared/git_atecon/naiveFC/python/src")

os.getcwd()

# Load additional packages/ functions
#=======================================
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sb
from naiveFC import *




# Open Data
#================
opt_date = 1           # SELECT (default = 1)

if opt_date==1:
    #pd.read_csv('beer.csv', index_col='obs', na_values=["NA"]) 
    df = pd.read_csv('beer.csv', na_values=["NA"]) 
    df.index = pd.period_range('1992-01', periods = len(df), freq="Q")
    del df["obs"]

elif opt_date==2:
    df = pd.read_csv('beer.csv', na_values=["NA"])   
    df.index = pd.to_datetime(df["obs"])
    del df["obs"]


"""
Call functions
"""

# Define object
#================
naive_obj = naiveFC(df.x)

# Call methods
#================
fc_meanf = naive_obj.meanf()
print("Mean FC:\n-------------")
print(fc_meanf)

naive_obj.use_median = True
fc_medianf = naive_obj.meanf()
print("Median FC:\n-------------")
print(fc_medianf)

naive_obj.use_median = False
fc_smeanf = naive_obj.smeanf()
print("Seasonal mean FC:\n-------------")
print(fc_smeanf)

naive_obj.use_median = True
fc_smedianf = naive_obj.smeanf()
print("Seasonal median FC:\n-------------")
print(fc_smedianf)


fc_ar1 = naive_obj.ar1f()
print("AR(1):\n-------------")
print(fc_ar1)

naive_obj.trend = True
fc_ar1trend = naive_obj.ar1f()
print("AR(1) + trend:\n-------------")
print(fc_ar1trend)

fc_rw = naive_obj.rwf()
print("RW:\n-------------")
print(fc_rw)

naive_obj.drift = True
fc_rwd = naive_obj.rwf()
print("RW + drift:\n-------------")
print(fc_rwd)

fc_snaive = naive_obj.snaive()
print("Seasonal naive FC:\n-----------------")
print(fc_snaive)


"""
if __name__=="__main__":
    main()
"""



