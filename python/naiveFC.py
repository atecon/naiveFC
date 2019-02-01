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
    """ Helper function printing error when CIs are demanded"""
    try:
        raise NoBootCi
    except NoBootCi:
        print("Error: Bootstrap confidence intervals are not supported, yet")
        print()


def gen_index(h):
    """ Helper function for generating rownames for the
    index of the fc-matrix """
    L = ["h="] * h
    for i in range(h):
        L[i] += str(i+1)
        
    return L


def gen_colname():
    """ Helper function constructing column name(s)
    NOTE: add CIs later
    """    
    
    return "point"

    
def series_mat_concat(y, m):    
    """
        Helper function concatenating series y with a vector m
        of the same length
        return matrix T by 2 matrix
    """
    # TODO: add a check that y and m have the same length    
    try:        
        #if y.shape[0] == m.flatten(order="F").shape[0]:
        # TODO: implement vec() equivalent!
        if y.shape[0] == m.shape[0]:
            my = np.asmatrix(y).transpose() # TODO: replace by vec()                        
            return np.concatenate((my, m), axis=1)
        else:
            raise UnequalLength        
    except UnequalLength:
        print("Series y and vector m are of different length")
        print()
        

def get_freq_as_vector(y):
    """
        Helper function obtain a series of the minor
        observations of y
    """        
    
    """ Determine the frequency of series 'y' """
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
    

def get_mean_obsminor(y, h, use_median=False):
    """
    Helper function obtain historical mean value for each separate
    quarter, month, or day across all years
        
    Parameters
    ----------
    y: series
        The dependent series of length T
    h: integer
        Compute up to the 'h' forecast horizon (default 10 periods)
    use_median: bool
        Compute median instead of mean (default False)
    --------
    Returns a pd by 1 matrix with mean/ median values where pd
    is the periodicity of y (quarterly: pd=4, monthly: pd=12,
    daily: pd=7 etc.)
    """
    
    # TODO: add a check that "y" has no freq-index (is a TS)
    
    # Concatenate series y with vector of minor frequency of y   
    out = series_mat_concat(y, get_freq_as_vector(y))
        
    df = pd.DataFrame(out, columns=["y", "freq"])
    df.index = y.index
    
    """
        Get mean-value for each minor frequency
        
        NOTE: it is assumed that at least 1 obs for every
        potential obsminor value exists in 'y'
        TODO: Given the NOTE, add a check and warning that seasonal-fc
        won't be available in this case -- at least for some freq.
    """
    if use_median==False:
        return df.groupby("freq").mean()
    else:
        return df.groupby("freq").median()

    
def my_ols(y,X):
    """
    Helper function estimating coefficients by OLS using linear algebra
    
    Parameters
    ----------
    y: series
        The dependent series of length T
    X: data frame
        data frame of regressors of dimension T by k
    --------
    Returns a k by 1 matrix    
    """
    
    try:    # Note: rcond=none works only for numpy 1.15.4 and later
        return np.linalg.lstsq(X,y,rcond=None)[0]  # 0=grab only coeff. matrix
    except:
        return np.linalg.lstsq(X,y)[0].ravel()
    
      

def meanf(y, h=10, level=90, fan=False, nboot=0, blength=4):
        
    """
    Compute mean forecasts of all future values which are equal to the
    mean of historical data
    Parameters
    ----------
    y: series
        The series which to forecast 
    h: integer
        Compute up to the 'h' forecast horizon (default 10 periods)
    level: scalar
        The 'alpha'-quantile level for computing the for 1-alpha forecast
        intervall (default 90 pct.) -- not implemented, yet!
    fan: bool
        Compute fan-chart (default FALSE) -- not implemented, yet!
    nboot: integer
        No. of bootstrap replications (default 0) -- not relevant, yet!
    blength: integer
        Mean length of block-bootstrap sample. If (nboot>0 && blength==0), run
        iid bootstrape; if (nboot>0 && blength>0) run block-bootstrap);
        (default = 0) -- not implemented, yet!
    --------
    Returns h by 1 series of forecast values    
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
    Compute seasonal mean forecasts which are equal to the season specific
    mean value based on historical data.    
    
    Parameters
    ----------
    y: series
        The series which to forecast 
    h: integer
        Compute up to the 'h' forecast horizon (default 10 periods)
    level: scalar
        The 'alpha'-quantile level for computing the for 1-alpha forecast
        intervall (default 90 pct.) -- not implemented, yet!
    fan: bool
        Compute fan-chart (default FALSE) -- not implemented, yet!
    nboot: integer
        No. of bootstrap replications (default 0) -- not relevant, yet!
    blength: integer
        Mean length of block-bootstrap sample. If (nboot>0 && blength==0), run
        iid bootstrape; if (nboot>0 && blength>0) run block-bootstrap);
        (default = 0) -- not implemented, yet!
    --------
    Returns h by 1 series of forecast values    
    """
      
    if nboot>0:
        print_noboot()
        return None
    
    elif nboot==0:
    
        """ obtain historical mean value for each separate freq """
        
        fmean = get_mean_obsminor(y, h)  # T by 2 data frame
        
        # last obsminor value, e.g. last quarter
        last = get_freq_as_vector(y)[-1]
      

        """ Re-order fc to make sure that 1st forecast
            corresponds to the right month, quarter etc. """
    pd = fmean.shape[0]:	# no. of periodicities
    if last<pd:
    
        selmat = np.zeros(pd,1)
        counter = last + 1
        loop i=1..max(values($obsminor)) -q
            selmat[counter] = $i
            counter = (counter==max(values($obsminor))) ? 1 : (counter+1)
        endloop
        fc = msortby(fc~selmat,2)[,1]
        
        

            #Reorder fmean according to 'last'
            #For instance, if last=Q3 then next is Q4 and then Q1 etc.
            #We construct a len(fmean) by 1 vector whose indices direct
            #to a sequence of the following obsminor frequencies                        
        print(fmean)
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
        """
        

def snaive(y, h=10, level=90, fan=False, nboot=0, blength=4):
    """
    Compute the seasonal naive forecasts which are equivalent to forecasts
    from ARIMA(0,0,0)(0,1,0) model.
        
    Parameters
    ----------
    y: series
        The series which to forecast 
    h: integer
        Compute up to the 'h' forecast horizon (default 10 periods)
    level: scalar
        The 'alpha'-quantile level for computing the for 1-alpha forecast
        intervall (default 90 pct.) -- not implemented, yet!
    fan: bool
        Compute fan-chart (default FALSE) -- not implemented, yet!
    nboot: integer
        No. of bootstrap replications (default 0) -- not relevant, yet!
    blength: integer
        Mean length of block-bootstrap sample. If (nboot>0 && blength==0), run
        iid bootstrape; if (nboot>0 && blength>0) run block-bootstrap);
        (default = 0) -- not implemented, yet!
    --------
    Returns h by 1 series of forecast values    
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
        # TODO: the following commented block doesn't work
        """        
        return pd.Series(fc[:,0],
                         index=gen_index(h),
                         name=gen_colname())
        """


def rwf (y, h=10, drift=False, level=90, fan=False, nboot=0, blength=4):
    """
    Compute random-walk forecasts which are equal to the last realization
    in case of no drift or to the last realization plus some historical mean
    growth rate per period, respectively.
        
    Parameters
    ----------
    y: series
        The series which to forecast
    h: integer
        Compute up to the 'h' forecast horizon (default 10 periods)
    drift: bool
        Add and consider historial mean growth rate (drift) (default False)
    level: scalar
        The 'alpha'-quantile level for computing the for 1-alpha forecast
        intervall (default 90 pct.) -- not implemented, yet!
    fan: bool
        Compute fan-chart (default FALSE) -- not implemented, yet!
    nboot: integer
        No. of bootstrap replications (default 0) -- not relevant, yet!
    blength: integer
        Mean length of block-bootstrap sample. If (nboot>0 && blength==0), run
        iid bootstrape; if (nboot>0 && blength>0) run block-bootstrap);
        (default = 0) -- not implemented, yet!
    --------
    Returns h by 1 series of forecast values    
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
    
    
def ar1f(y, h=10, const=True, trend=False, level=90,
          fan=False, nboot=0, blength=4):
    
    """
    Compute AR(1)-based forecast with or without linear trend.    
        
    Parameters
    ----------
    y: series
        The series which to forecast
    h: integer
        Compute up to the 'h' forecast horizon (default 10 periods)
    trend: bool
        Add and consider linear trend as additional regressor
    level: scalar
        The 'alpha'-quantile level for computing the for 1-alpha forecast
        intervall (default 90 pct.) -- not implemented, yet!
    fan: bool
        Compute fan-chart (default FALSE) -- not implemented, yet!
    nboot: integer
        No. of bootstrap replications (default 0) -- not relevant, yet!
    blength: integer
        Mean length of block-bootstrap sample. If (nboot>0 && blength==0), run
        iid bootstrape; if (nboot>0 && blength>0) run block-bootstrap);
        (default = 0) -- not implemented, yet!
    --------
    Returns h by 1 series of forecast values    
    """
    

    if nboot>0:
        print_noboot()
        return None
    
    else:
        T = y.shape[0]        
        Y = pd.DataFrame({'y': y})
                
        # Add intercept/ trend
        if const:            
            Y["const"] = 1        
        if trend:
            Y["time"] = pd.Series(1*np.ones(T).cumsum(), index=Y.index)
            
        # Add 1st lag
        Y["Y_1"] = Y.iloc[:,0].shift(1)     # y~const~trend~y(-1)
        Y.dropna(axis=0, inplace=True)
                
        # OLS        
        y = Y.iloc[:,0]
        X = Y.iloc[:,1:]    # const~(trend)~y(-1)
        bhat = my_ols(y,X)        

        # Iterative forecast
        fc = recfc(Y,bhat,h,const,trend)

        print(fc)
        
        # Finalize: add row and column strings to series        
        return pd.Series(fc[:,0], index=gen_index(h), name=gen_colname())
    
    
def recfc(Y,bhat,h,const,trend):
    """
    Helper function for constructing h-step ahead iterative forecasts
    based on AR(1) model.
    
    Parameters
    ----------
    Y: data frame
        Y has size T by (1+k) with elements y~const~trend~y(-1)
    bhat: matrix
        k by 1 vector of (OLS-based) point estimates
    h: integer
        Compute up to the 'h' forecast horizon (default 10 periods)
    const: bool
        Estimated model with intercept
    trend: bool
        Estimated model with linear trend
    --------
    Returns h by 1 series of forecast values
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



# TODO: add a wrapper to estimate all models at once and combine FCs
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