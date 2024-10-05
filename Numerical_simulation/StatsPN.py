# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:09:39 2021

@author: aikch

class StatsPN that calculates the true positive (TP), FP, TN, FN and recall, precision and other statistical measures from a prediction matrix input. 

"""

#%% Official libraries

import numpy as np

#%% Class

class StatsPN(object):
    
    
    def __init__(self, arry ):
        """Get the statstics from count matrix
        Column (1st dimension) - Truth
        Row (2nd dimension) - Prediction
        """
        #Save the array
        self.arry = arry
        
        #Array length in Truth axis
        self.lens = np.size(self.arry,0)
        #Array length in Pred axis
        self.lensP = np.size(self.arry,1)
        
        #Initialize stats. basis array
        #True Positive
        self.TP = np.zeros( (self.lens,1) )
        #False Positive
        self.FP = np.zeros( (self.lensP,1) )
        #True Negative
        self.TN = np.zeros( (self.lens,1) )
        #False Negative
        self.FN = np.zeros( (self.lens,1) )
        
        #Initialize statisticals arrays
        #Positive Predicitive Value, Precision
        self.precision = np.zeros( (self.lens,1) )
        #True Positive Rate, Recall
        self.recall = np.zeros( (self.lens,1) )
        #True negative rate, selectivity
        self.selectivity = np.zeros( (self.lens,1) )
        #Negative Prediction Value
        self.NPV = np.zeros( (self.lens,1) )
        
        #Derived Statistical Metrics
        #False Omission Rate
        self.FOR = np.zeros( (self.lens,1) )
        #False Positive Rate
        self.FPR = np.zeros( (self.lens,1) )
        
        #Prediction Probability (Old)
        self.PredProb = np.zeros( (self.lens \
                      , self.lensP) )
        
        
        #The iteration arrays for LOOP
        #For truth axis
        self.ItT = np.arange(self.lens)
        #For Pred axis
        self.ItP = np.arange(self.lensP)        
        
        
    def CalcBasis(self):
        """Calculate the TP,FP,TN,FN
        .See LabNotebook book 6, pg 11"""

        
        #True Positive
        for ii in self.ItT:
            self.TP[ii] = self.arry[ii,ii]
        #False Positive
        for ii in self.ItT:
            for jj in self.ItP:
                #Sum over row exc. ii = jj
                if jj != ii:  
                    self.FN[ii] \
                        += self.arry[ii,jj] 
        #False Negative
        for jj in self.ItP:
            for ii in self.ItT:
                #Sum over column exc. ii = jj
                if ii != jj:
                    self.FP[jj] \
                        += self.arry[ii,jj]
        #True Negative
        for kk in self.ItT:
            for ii in self.ItT:
                for jj in self.ItP:
                    #Sum over off axis 
                    if ii != kk and jj != kk:
                        self.TN[kk] += \
                            self.arry[ii,jj]
                            
    def CalcMetrics(self):
        """Calculate Metrics such as Precision and Recall"""
        #Calculate primary Metrics
        for kk in self.ItT:
            self.precision[kk] = self.TP[kk] \
                / (self.TP[kk] + self.FP[kk])
            self.recall[kk] = self.TP[kk] \
                / (self.TP[kk] + self.FN[kk])
            self.selectivity[kk] = self.TN[kk] \
                / (self.TN[kk] + self.FP[kk])
            self.NPV[kk] = self.TN[kk] \
                / (self.TN[kk] + self.FN[kk])  
                
        #Calculate secondary Metrics                
        self.FOR = 1 - self.NPV
        self.FPR = 1 - self.selectivity
    def CalcProbMat(self):
        """Calculate Probability Matrix (presented in same way as before"""
        for ii in self.ItT:
            RowSum = np.sum( \
                    self.arry[ii,:] )
            RowNorm = self.arry[ii,:] / RowSum
            self.PredProb[ii,:] = RowNorm
            
        
        
        

