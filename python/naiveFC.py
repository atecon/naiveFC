#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 10:52:58 2019

@author: Artur Tarassow
"""


#=========================================================
    
import pandas as pd
import numpy as np

def print_noboot():
    """ Print Error"""
    # TODO: look for funcerr
    print("Error: Bootstrap confidence intervals are not supported, yet")

def gen_index(h):
    """ Construct list of rownames for the index of the fc-matrix """
    L = ["h="] * h
    for i in range(h):
        L[i] += str(i+1)        
    return L

def gen_colname():
    """ construct column names """
    return "point"   

    
def series_mat_concat(y, m):    
    """
        concatenate series y with a vector m
        of the same length
        return matrix T by 2 matrix
    """
    # TODO: add a check that y and m have the same length    
    my = np.asmatrix(y).transpose() # TODO: vec() rather t. transpose()
    return np.concatenate((my, m), axis=1)


def get_freq_as_vector(y):
    """
        Obtain a series of the minor observations
        of y
    """        
    
    """ Determine the frequency of series 'y' """
    # TODO: add a check whether y.index has TS-structure    
    f = y.index.freqstr # string indicating frequency (A, Q; M, D)    
        
    if f.find("Q",0,1) is not -1: # quarterly
        return np.asmatrix(y.index.quarter).transpose()        
    elif f.find("M",0,1) is not -1: # monthly
        return np.asmatrix(y.index.month).transpose()
    elif f.find("d",0,1) is not -1: # daily
        return np.asmatrix(y.index.day).transpose()
    elif f.find("A",0,1) is not -1:
        print("You cannot apply the choosen method to annual data.")
        return -1
    

def get_mean_obsminor(y, h):
    """
    obtain historical mean value for each separate quarter, month,
    or day across all years
    """
    
    # Concatenate series y with vector of minor frequency of y
    m = series_mat_concat(y, get_freq_as_vector(y))

    if m is not -1:
        df = pd.DataFrame(m, columns=["y", "freq"])
        df.index = y.index
    
        """
        get mean-value for each minor frequency        
        NOTE: it is assumed that at least 1 obs for every
        potential obsminor value exists in y
        TODO: add a check and warning in future that seasonal-fc
        won't be available in this case -- at least for some freq.
        """
        fmean = df.groupby("freq").mean()       # TODO: add median()
        return fmean
    
      
    
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
        fc = pd.Series(np.ones((h)),
                       index=gen_index(h),
                       name=gen_colname()) * np.mean(y)        
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
    
    elif nboot==0:
    
        """ obtain historical mean value for each separate freq """
        fmean = get_mean_obsminor(y, h)  # T by 2 data frame                
        last = fmean.index[-1]      # last obsminor value, e.g. last quarter
        
        # Reorder fmean according to 'last'
        # For instance, if last=Q3 then next is Q4 and then Q1 etc.
        # We construct a len(fmean) by 1 vector whose indices direct
        # to a sequence of the following obsminor frequencies        
        fc = np.zeros((max(h,len(fmean)),1))
                        
        for i in range(len(fc)):
            
            if (last)==len(fmean):
                last = 0           # back to 1st obsminor value
                                        
            fc[i] = fmean["y"][last+1]  # fmean.index is 1-based
                        
            last=last+1       # update

        # Finalize: add row and column strings to series        
        return pd.Series(fc[:,0],
                         index=gen_index(h),
                         name=gen_colname())  


def snaive(y, h=10, level=90, fan=False, nboot=0, blength=4):
    """
    Returns forecasts and prediction intervals from an
    ARIMA(0,0,0)(0,1,0)
    model where m is the seasonal period.
    """
    
    if nboot>0:
        print_noboot()
        return None
    if nboot==0:
        # Concatenate series y with vector of minor frequency of y
        # return T by 2 vector 'm'
        m = series_mat_concat(y, get_freq_as_vector(y))

        # read periodicity per year (A=1, Q=4, ..)
        freq = m[:,1].max()

        # read last 'freq' obsminor values, e.g. last 4 quarters, etc.
        fc = m[-min(h,freq):,0]
       
        if h<=freq:            
            return pd.Series(fc[-h:,0],
                             index=gen_index(h),
                             name=gen_colname())
        else:
            for i in range(h-freq): # fill up remaining horizons
                fcnew = fc[-freq]                
                fc = np.concatenate((fc,fcnew))
                
        # Finalize: add row and column strings to series        
        return fc
        """
        return pd.Series(fc[:,0],
                         index=gen_index(h),
                         name=gen_colname())
        """


def rwf (y, h=10, drift=False, level=90, fan=False, nboot=0, blength=4):
    """
    Random-walk forecast of all future values are equal to the
    last historical data point.
    Random-walk w. dtrift and forecast of all future values are
    equal to the last historical data point.
    Returns forecasts (and prediction intervals for an iid model)
    applied to y  
    """    
    if nboot>0:
        print_noboot()
        return None
    
    else:                
        fc = np.ones(h) * y[-1]
        
        if drift:        
            # equivalent to an ARIMA(0,1,0) model with an
            # optional drift coefficient
            g = (y[-1]-y[0])/(np.shape(y)[0]-1)  # avg. growth rate (drift)        
            fc = fc + np.cumsum(np.ones(h)) * g            
            
        # Finalize: add row and column strings to series        
        return pd.Series(fc, index=gen_index(h), name=gen_colname())


def my_ols(y,X):
    """
    Estimate params. by OLS using linear algebra
    """
    return np.linalg.lstsq(X,y,rcond=None)[0] # 0=grab only coeff. matrix

    
def ar1f(y, h=10, const=True, trend=False, level=90,
          fan=False, nboot=0, blength=4):
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
        fc = recfc(Y,bhat,h,const,trend)

        # Finalize: add row and column strings to series        
        return pd.Series(fc[:,0], index=gen_index(h), name=gen_colname())
    
    
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
            
    return fc




#def avgfc(y, h=10, const=True, trend=False, level=90,
#          fan=False, nboot=0, blength=4):
#    
#    """
#    Computes the mean (average) and cross-sectional (across forecast methods)
#    standard deviation at each horizon using all simple forecast methods
#    available.
#    """
#
#    if nboot>0:
#            print_noboot()
#            return None      
#    else:
#
#        # Base models
#        """
#        (1) Define a list of base model:
#            defarray("meanf(y,h)", "rwf(y,h)", "rwf(y,h,1)", "ar1f(y,h)")
#        (2) if $pd>1	# for seasonal data only
#            += "smeanf(y,h)"
#            += "snaive(y,h)"
#        (3) Use feval()
#        """
#    
#    
#        fc = np.matrix(zeros(h,nelem(M))
#        
#    loop i=1..nelem(M) -q
#        string s = sprintf("%s", M[i])
#        fc[,i] = @s[,1]	# only point-fc
#    endloop
#    fc = meanr(fc) ~ sdc(fc', rows(fc')-1)' ~ fc
#    if $pd == 1
#        cnameset(fc, strsplit("average-fc sd meanf rwf rwf+drift AR(1)", " "))
#    else
#        cnameset(fc, strsplit("average-fc sd meanf rwf rwf+drift AR(1) smean snaive", " "))
#    endif
#    rnameset(fc, rownam(h))
#    return fc
#end function