#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 10:52:58 2019

@author: at
"""

# Set working dir
import os
os.getcwd()
# ADJUST THIS PATH ACCORDING TO YOUR MACHINE
os.chdir("/home/at/git/python_exercises/src/naiveFC")
os.getcwd()

# Load some packages/ functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Read-in raw data into Pandas dataframe
# Note: the dataset doesn't seem to include missings
df = pd.read_csv('beer.csv', na_values=["NA"]) 

# Assign date string to index
df.index = pd.to_datetime(df['obs'])
print(df.index)
print(df.index.dtype)


def print_noboot():
    """ Print Error"""
    # TODO: look for funcerr
    print("Error: Bootstrap confidence intervals are not supported, yet")

def gen_index(h):
    """ Construct list of rownames for index """
    L = ["h="] * h
    for i in range(h):
        L[i] += str(i+1)        
    return L
    
def gen_colname():
    """ construct column names """
    return "point"   

def meanf(y, h=10, level=90, fan=False, nboot=0, blength=4):
    """
    Mean forecasts of all future values are equal to the
    mean of historical data
    Returns forecasts (and prediction intervals for an iid model)
    applied to y
    """    
    if nboot>0:
        print_noboot()
        return None
    if nboot==0:
        fc = pd.Series(np.ones((h)),gen_index(h)) * np.mean(y)        
        fc.name = gen_colname()
        return fc
    


def rwf (y, h=10, drift=False, level=90, fan=False, nboot=0, blength=4):
    """
    Random-walk forecast of all future values are equal to the
    last historical data point.
    Random-walk w. dtrift and forecast of all future values are equal to the
    last historical data point.
    Returns forecasts (and prediction intervals for an iid model)
    applied to y  
    """    
    if nboot>0:
        print_noboot()
        return None
    
    else:                
        fc = pd.Series(np.ones(h),gen_index(h)) * y[-1]
        
        if drift:        
            # equivalent to an ARIMA(0,1,0) model with an optional drift coefficient
            g = (y[-1]-y[0])/(np.shape(y)[0]-1)     # average growth rate (drift)        
            fc = fc + np.cumsum(np.ones(h)) * g            
            
        fc.name = gen_colname()
        return fc

def my_ols(y,X):
    return np.linalg.lstsq(X,y,rcond=None)[0] # 0=only coeff. matrix

    
def ar1f (y, h=10, const=True, trend=True, level=90, fan=False, nboot=0, blength=4):
    """
    AR(1) forecast with or withou linear trend.
    Returns forecasts (and prediction intervals for an iid model)
    applied to y  
    """
    if nboot>0:
        print_noboot()
        return None
    
    else:
        T = df.shape[0]        
        Y = pd.DataFrame(df["x"])
        
        # Add intercept
        if const:            
            Y["const"] = 1        
        if trend:
            Y["time"] = pd.Series(1*np.ones(T).cumsum(), index=Y.index)
            
        # Add 1st lag
        Y["Y_1"] = Y.iloc[:,0].shift(1)
        Y.dropna(axis=0, inplace=True)
        
        # OLS        
        y = Y.iloc[:,0]
        X = Y.iloc[:,1:]
        bhat = my_ols(y,X)
                
        # Iterative forecast
        return recfc(Y,bhat,h,const,trend)

        
def recfc(y,bhat,h,const,trend):
    """
    Construct h-step ahead iterative forecast based on AR(1)
    """
    
    fc = np.zeros((h,1))
    m = y.iloc[:,-1]    # grab last obs.
    
    if const:
        fc[0] = bhat[0]    
    if trend: 
        fc[0] += bhat[1]*m[1] + bhat[2]*m[2]
    else:
         fc[0] += bhat[1]*m[1]            

    # Start the recursion
    for i in range(1,h):
        if const:
            fc[i] = bhat[0]    
        if trend: 
#            print((m[2]+1))
            fc[i] += bhat[1]*(m[2]+1) + bhat[2]*fc[i-1]
        else:
            fc[i] += bhat[1]*fc[i-1]
            
    return fc



fc_ar1 = ar1f(df["x"])
print(fc_ar1)

    
# =============================================================================
# # CALL
# =============================================================================
fc_meanf = meanf(df["x"])
print(fc_meanf)

fc_rw = rwf(df["x"])
print(fc_rw)

fc_rwd = rwf(df["x"], drift=True)
print(fc_rwd)

fc_ar1 = ar1f(df["x"])
