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
print(fc_meanf)

fc_smeanf = naive_obj.smeanf()
print(fc_smeanf)

fc_ar1 = naive_obj.ar1f()
print(fc_ar1)

naive_obj.trend = True
fc_ar1trend = naive_obj.ar1f()
print(fc_ar1trend)

fc_rw = naive_obj.rwf()
print(fc_rw)

naive_obj.drift = True
fc_rwd = naive_obj.rwf()
print(fc_rwd)

fc_snaive = naive_obj.snaive()
print(fc_snaive)


"""
if __name__=="__main__":
    main()
"""



