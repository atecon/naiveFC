#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 10:52:58 2019

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
import matplotlib.pyplot as plt
import seaborn as sb

"""
Read-in raw data into Pandas dataframe
"""
opt_date = 1        # SELECT

if opt_date==1:
    #pd.read_csv('beer.csv', index_col='obs', na_values=["NA"]) 
    df = pd.read_csv('beer.csv', na_values=["NA"]) 
    df.index = pd.period_range('1992-01', periods = len(df), freq="Q")
    del df["obs"]
elif opt_date==2:
    df = pd.read_csv('beer.csv', na_values=["NA"])   
    df.index = pd.to_datetime(df["obs"])
    del df["obs"]



#df.groupby(pd.Grouper(freq='Q')).mean()  # update for v0.21+



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
    
    
def smeanf(y, h=10, level=90, fan=False, nboot=0, blength=4):
    """
    Forecasts of all future values are equal to the mean of
    specific seasons based on historical data.    
    Returns forecasts (and prediction intervals for an iid model)
    applied to y
    """    
    if nboot>0:
        print_noboot()
        return None
    if nboot==0:
    
        # print frequency
        period = y.index.freqstr
        print(period)
    
        # Once we have the frequency, we can construct a periodicity series
        #if period.find('Q'):        
        if period.startswith('A'):
           y["period"] = period.startswith('A')
        elif period.startswith('Q'):
            y["period"] = period.startswith('Q')
        elif period.startswith('M'):
            y["period"] = period.startswith('M')
        elif period.startswith('D'):
            y["period"] = period.startswith('D')
    
        print(y)
        
fc_smeanf = smeanf(df["x"])
#print(fc_smeanf)


# %%

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
    """
    Estimate params. by OLS using linear algebra
    """
    return np.linalg.lstsq(X,y,rcond=None)[0] # 0=grab only coeff. matrix

    
def ar1f (y, h=10, const=True, trend=False, level=90, fan=False, nboot=0, blength=4):
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
        Y["Y_1"] = Y.iloc[:,0].shift(1)     # y~const~trend~y(-1)
        Y.dropna(axis=0, inplace=True)
        
        # OLS        
        y = Y.iloc[:,0]
        X = Y.iloc[:,1:]    # const~trend~y(-1)
        bhat = my_ols(y,X)
                
        # Iterative forecast
        return recfc(Y,bhat,h,const,trend)

        
def recfc(Y,bhat,h,const,trend):
    """
    Construct h-step ahead iterative forecast based on AR(1)
    Y:  data frame, T by (1+k) with elements y~const~trend~y(-1)
    """
        
    fc = np.zeros((h,1))
    m = np.asmatrix(Y.iloc[-1,:-1])   # 1 by (1+k); grab last obs. row
    # position y(t) at last position    
    m = np.concatenate((m[:,1:], m[:,0]), axis=1)   # const~trend~y(t)
    bhat = np.asmatrix(bhat)       # 1 by k

    # 1-step ahead forecast     
    fc[0] = m * np.transpose(bhat)
    
    if h>1:
        # Start the recursion for 2-step-ahead forecast
        for i in range(1,h):
            
            # replace y(t) by last forecast value
            m[:,-1] = fc[i-1]   
            
            if trend: 
                # update the linear trend value
                m[:,1] += 1
                
            # compute new point forecast
            fc[i] = m * np.transpose(bhat)
            
    return pd.Series(fc[:,0])




## %%
    
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
print(fc_ar1)

