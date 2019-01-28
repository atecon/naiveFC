#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    *** Sample script for running the naiveFC package ***


Created on Fri Jan 25 19:25:19 2019

@author: Artur Tarassow
"""


# Set working dir
import os
os.getcwd()
# ADJUST THIS PATH ACCORDING TO YOUR MACHINE
os.chdir("/home/at/git/naiveFC/python")
os.getcwd()

# Load some packages/ functions
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sb

# import naiveFC

"""
Open Data
"""
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

fc_meanf = meanf(df["x"])
print(fc_meanf)

fc_smeanf = smeanf(df["x"])
print(fc_smeanf)

fc_ar1 = ar1f(df["x"])
print(fc_ar1)

fc_ar1trend = ar1f(df["x"], trend=True)
print(fc_ar1trend)

fc_rw = rwf(df["x"])
print(fc_rw)

fc_rwd = rwf(df["x"], drift=True)
print(fc_rwd)

fc_snaive = snaive(df["x"])
print(fc_snaive)