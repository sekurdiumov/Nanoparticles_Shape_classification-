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
    a = np.arange(2, size[i] + 1, 1)
    b = np.zeros_like(a)
    szz = np.zeros_like(a)
    for j in range(len(a)):
    
        b[j] = int(np.sqrt((size[i])**2 - (a[j])**2))
        
        if b[j] == 0:
            b[j] = 2
        
        szz[j] = size[i]
        
        #if a[j] >= 10 and b[j] >= 10:
            
        a0 = np.concatenate((a0, np.array([a[j]])), axis = 0)    
        b0 = np.concatenate((b0, np.array([b[j]])), axis = 0)
        a0 = np.concatenate((a0, np.array([b[j]])), axis = 0)    
        b0 = np.concatenate((b0, np.array([a[j]])), axis = 0)
        sz = np.concatenate((sz, np.array([szz[j]])), axis = 0)
        sz = np.concatenate((sz, np.array([szz[j]])), axis = 0)
    
a = a/PWL
b = b/PWL

sz = sz/PWL
a0 = a0/PWL
b0 = b0/PWL
           
            
zigzags = np.zeros((len(a0), 5))
zigzags[:,0] = 4
zigzags[:,1] = sz
zigzags[:,2] = a0/2
zigzags[:,3] = b0/2

zigzags = np.unique(zigzags, axis = 0)

for i in range(len(zigzags)):
    
    if zigzags[i,2] < 5/PWL or zigzags[i,3] < 5/PWL:
        zigzags[i,4] = 1

zigzags_good = zigzags[zigzags[:,4] == 0]
zigzags_bad = zigzags[zigzags[:,4] == 1]

fname = 'zigzags_good_exp_data.mat'
fname1 = 'zigzags_bad_exp_data.mat'

scio.savemat(fname, {'data' : zigzags_good[1:,:]})
scio.savemat(fname1, {'data' : zigzags_bad[1:,:]})