#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 10:52:58 2019

@author: Artur Tarassow
"""


#=========================================================
    
import pandas as pd
import numpy as np
import math as ma

class naiveFC(object):
    
    def __init__(self, y):
        """Return an fcstats object"""
        self.y = pd.Series(y)              # series of realizations        
        self.Ty = y.shape[0]
        self.h = 10                     # forecast horizon
        self.level = 90                 # width of forecast interval
        self.fan = False                # compute fan chart
        self.nboot = 0                  # no. of bootstrap iterations
        self.blength = 4                # block-length for stationary block-bootstrap
        self.const = True               # add intercept to ar(1) model
        self.trend = False              # add linear trend to ar(1) model
        self.drift = False              # add drift to rw model


    def print_noboot(self):
        """ Helper function printing error when CIs are demanded"""
        try:
            raise NoBootCi
        except NoBootCi:
            print("Error: Bootstrap confidence intervals are not supported, yet")
            print()

    def gen_index(self):
        """ Helper function for generating rownames for the
        index of the fc-matrix """
        L = ["h="] * self.h
        for i in range(self.h):
            L[i] += str(i+1)

        return L
    
    def gen_colname(self):
        """ Helper function constructing column name(s)
        NOTE: add CIs later
        """
        return "point"
    
        
    def series_mat_concat(self, m):
        """
            Helper function concatenating series y with a vector m
            of the same length
            return matrix T by 2 matrix
        """
        # TODO: add a check that y and m have the same length
        try:
            #if y.shape[0] == m.flatten(order="F").shape[0]:        
            if self.y.shape[0] == m.shape[0]:
                my = np.asmatrix(self.y).ravel().T            
                return np.concatenate((my, m), axis=1)
            else:
                raise UnequalLength        
        except UnequalLength:
            print("Series y and vector m are of different length")
            print()
            
    
    def get_freq_as_vector(self):
        """
            Helper function obtain a series of the minor
            observations of y
        """        
        
        """ Determine the frequency of series 'y' """
        f = self.y.index.freqstr # string indicating frequency (A, Q; M, D)    
            
        if f.find("Q",0,1) is not -1: # quarterly
            return np.asmatrix(self.y.index.quarter).ravel().T #transpose()        
        elif f.find("M",0,1) is not -1: # monthly
            return np.asmatrix(self.y.index.month).ravel().T
        elif f.find("d",0,1) is not -1: # daily
            return np.asmatrix(self.y.index.day).ravel().T
        elif f.find("A",0,1) is not -1:
            print("You cannot apply the choosen method to annual data.")
            return -1
        
    
    def get_mean_obsminor(self, use_median=False):
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
        out = self.series_mat_concat(self.get_freq_as_vector())
        df = pd.DataFrame(out, columns=["y", "freq"])
        df.index = self.y.index
        
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
    
        
    def my_ols(self, y,X):
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


    def recfc(self,Y,bhat):
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
            
            fc = np.zeros((self.h,1))
            m = np.asmatrix(Y.iloc[-1,:-1])   # 1 by (1+k); grab last obs. row
            # position y(t) at last position    
            m = np.concatenate((m[:,1:], m[:,0]), axis=1)   # const~trend~y(t)
            bhat = np.asmatrix(bhat)       # 1 by k
        
            # 1-step ahead forecast     
            fc[0] = m * np.transpose(bhat)
    
            if self.h>1:
                # Start the recursion for 2-step-ahead forecast
                for i in range(1,self.h):
                    
                    # replace y(t) by last forecast value
                    m[:,-1] = fc[i-1]   
                    
                    if self.trend:
                        # update the linear trend value
                        m[:,1] += 1
                        
                    # compute new point forecast
                    fc[i] = m * np.transpose(bhat)
    
            return fc



    def return_fc_as_series(self, fc):
        """Transform vector into series with proper row-/ column names"""
        
        return pd.Series(fc[0:self.h],
                  index=self.gen_index(),
                  name=self.gen_colname())        
        
    
    def meanf(self):
            
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
        
        if self.nboot>0:
            self.print_noboot()
            return None
        else:
            return pd.Series(np.ones((self.h)),
                           index = self.gen_index(),
                           name = self.gen_colname()) * np.mean(self.y)            
    
        
    def smeanf(self):
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
          
        if self.nboot>0:
            print_noboot()
            return None
        
        else:        
            """ obtain historical mean value for each separate freq """
            
            fmean = self.get_mean_obsminor()  # T by 2 data frame
            
            # last obsminor value, e.g. last quarter
            pd_last = np.asscalar(self.get_freq_as_vector()[-1]) # starts at '1'          
    
            """ Re-order fc to make sure that 1st forecast
                corresponds to the right month, quarter etc.
                For instance, if last=Q3 then next is Q4 and then Q1 etc.
                We construct a len(fmean) by 1 vector whose indices direct
                to a sequence of the following obsminor frequencies      
            """
            pd = fmean.shape[0]    # no. of periodicities (4=quarterly)
            
            if pd_last <= pd:                 # FIXME: acually '<'
                
                selmat = np.zeros((pd,1))     # selection matrix            
                counter = pd_last
                
                for i in range(0,pd):
                    if counter==pd:
                        counter = 1
                    else:
                        counter = counter + 1

                    selmat[counter-1] = 1+i     # '-1' -> consider "0-counting"
                    #print(selmat)
                    #print(counter)
                    
            fc = np.concatenate((fmean, selmat), axis=1)
            fc = np.asmatrix( fc[fc[:, 1].argsort()][:,0] ).T
                    
            # construct h-step ahead forecasts
            fc_r = fc.shape[0]        
            k = ma.ceil(self.h / fc_r)     # no. of necessary stackings
            
            fc = np.multiply( np.ones((fc_r,k)), fc ).flatten('F').T # column-vector
            
            return fc[0:self.h]
            
            # Finalize: add row and column strings to series                
            # BUG: can't transform fc to series
            """
            return self.return_fc_as_series(fc)
            """
            
    
    def snaive(self):
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
        
        if self.nboot>0:
            print_noboot()
            return None
        else:
            # Concatenate series y with vector of minor frequency of y
            # return T by 2 vector 'm'
            m = self.series_mat_concat(self.get_freq_as_vector())
            # BUG: series_mat_concat() or get_freq_as_vector() results in 
            # integer values for 'm'
            #print(type(m))
    
            # read periodicity per year (A=1, Q=4, ..)
            freq = m[:,1].max()
    
            # read last 'freq' obsminor values, e.g. last 4 quarters, etc.
            fc = m[-min(self.h, freq):,0]
           
            if self.h<=freq:            
                return pd.Series(fc[-self.h:,0],
                                 index = self.gen_index(),
                                 name = self.gen_colname())
            else:
                for i in range(self.h-freq): # fill up remaining horizons
                    fcnew = fc[-freq]
                    fc = np.concatenate((fc, fcnew))
                    
            # Finalize: add row and column strings to series        
            return fc[:self.h]
        
            # TODO: the following commented block doesn't work
            # BUG: can't transform fc to series            
            """        
                return self.return_fc_as_series(fc)
            """
    
    
    def rwf(self):
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
    
        if self.nboot>0:
            print_noboot()
            return None
        
        else:                
            fc = np.ones(self.h) * self.y[-1]
            
            if self.drift:
                # equivalent to an ARIMA(0,1,0) model with an
                # optional drift coefficient
                g = (self.y[-1] - self.y[0])/(np.shape(self.y)[0]-1)  # avg. growth rate (drift)        
                fc = fc + np.cumsum(np.ones(self.h)) * g        

            return self.return_fc_as_series(fc)

        
    def ar1f(self):
        
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
        
    
        if self.nboot>0:
            print_noboot()
            return None
        
        else:
            T = self.y.shape[0]        
            Y = pd.DataFrame({'y': self.y})
                    
            # Add intercept/ trend
            if self.const:
                Y["const"] = 1
            if self.trend:
                Y["time"] = pd.Series(1*np.ones(T).cumsum(), index=Y.index)
                
            # Add 1st lag
            Y["Y_1"] = Y.iloc[:,0].shift(1)     # y~const~trend~y(-1)
            Y.dropna(axis=0, inplace=True)
                    
            # OLS        
            y = Y.iloc[:,0]
            X = Y.iloc[:,1:]    # const~(trend)~y(-1)
            bhat = self.my_ols(y,X)
    
            # Iterative forecast
            fc = self.recfc(Y, bhat)            
               
            # Finalize: add row and column strings to series        
            return fc
    
            # TODO: doesn't work, yet!
            """return self.return_fc_as_series(fc[:,0])
            """
        
    
    
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
    #        (2) if $pd>1    # for seasonal data only
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
    #        fc[,i] = @s[,1]    # only point-fc
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