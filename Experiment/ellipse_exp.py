# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:53:15 2022

@author: sk1u19
"""
import numpy as np
import scipy.io as scio

PWL = 64
size = np.arange(10, 33, 1)
a0 = np.array([0])
b0 = np.array([0])
sz = np.array([0])

for i in range(len(size)):
    b = np.arange(2, size[i]+1, 2)
    a = np.zeros_like(b)
    szz = np.zeros_like(a)
    for j in range(len(a)):
    
        a[j] = int(size[i])    
        szz[j] = size[i]
    
    a0 = np.concatenate((a0, a), axis = 0)  
    a0 = np.concatenate((a0, b), axis = 0)  
    b0 = np.concatenate((b0, b), axis = 0)
    b0 = np.concatenate((b0, a), axis = 0)
    sz = np.concatenate((sz, szz), axis = 0)
    sz = np.concatenate((sz, szz), axis = 0)
    
a = a/PWL
b = b/PWL

sz = sz/PWL
a0 = a0/PWL
b0 = b0/PWL
           
            
ellipses = np.zeros((len(a0), 5))
ellipses[:,0] = 2
ellipses[:,1] = sz
ellipses[:,2] = a0/2
ellipses[:,3] = b0/2

ellipses = np.unique(ellipses, axis = 0)

for i in range(len(ellipses)):
    
    if ellipses[i,2] < 5/PWL or ellipses[i,3] < 5/PWL:
        ellipses[i,4] = 1

ellipses_good = ellipses[ellipses[:,4] == 0]
ellipses_bad = ellipses[ellipses[:,4] == 1]

fname = 'ellipses_good_exp_data.mat'
fname1 = 'ellipses_bad_exp_data.mat'

scio.savemat(fname, {'data' : ellipses_good})
scio.savemat(fname1, {'data' : ellipses_bad})