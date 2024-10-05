# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:19:17 2023

@author: sk1u19
"""
import numpy as np
import scipy.io as scio
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

plt.style.use('dark_background') 
#plt.style.use('default') 
matplotlib.rcParams.update({'font.size' : 25})

## Evaluating correlation coefficient for real maps:
def eval_Rxy(A, B):
    meanA = np.mean(A)
    meanB = np.mean(B)
    cov = np.mean((A - meanA)*(B - meanB))
    stdA = np.std(A)
    stdB = np.std(B)
    Rxy = cov/(stdA * stdB)
    
    return Rxy

def load_and_norm(fname):
    M1 = scio.loadmat(fname)
    E = M1['E'].astype(float)
    E = E/np.max(E)
    
    return E

## Plotting:
def plot(A):
    
    x = np.linspace(-256, 256, 512)*10.8/640
    
    plt.figure(figsize = [7, 6])
    plt.pcolormesh(x, x, A, cmap = 'hot')
    plt.colorbar()
    #plt.clim(0, 1)
    plt.xlabel('x, in λ')
    plt.ylabel('y, in λ')
#%% Select the data:
ii = np.zeros(1440)
jj = np.zeros(1440)
kk = np.zeros(1440)
truth = np.zeros(1440)
sz = np.zeros(1440)
aa = np.zeros(1440)
bb = np.zeros(1440)

counter = 0

for k in range(40):
    
    fname = 'GT_D' + str(k) + '.mat'
    M1 = scio.loadmat(fname)
    
    label = M1['GT']
    size = M1['size']
    a = M1['a']
    b = M1['b']
    
    for n in range(36):
        
        i = int(n/6)
        j = int(n%6)
        
        if label[0,n] >= 2 and label[0,n] <= 4:
            a[0,n] = 2*a[0,n]
            b[0,n] = 2*b[0,n]
        
        ii[counter] = i
        jj[counter] = j
        kk[counter] = k
        truth[counter] = label[0,n]
        sz[counter] = size[0,n]
        aa[counter] = a[0,n]
        bb[counter] = b[0,n]
        
        counter = counter + 1
        
dictt = {'k' : kk,
         'i' : ii,
         'j' : jj,
         'Ground Truth' : truth,
         'size' : sz,
         'a' : aa,
         'b' : bb}

df = pd.DataFrame(dictt)

df_ellipse = df[df['Ground Truth'] == 2]
df_circ = df_ellipse[np.abs(df_ellipse['a'] - df_ellipse['b']) < 0.02]

df_rect = df[df['Ground Truth'] == 1]
df_sqr = df_rect[np.abs(df_rect['a'] - df_rect['b']) < 0.02]

df_tri = df[df['Ground Truth'] == 0]
df_z = df[df['Ground Truth'] == 3]
df_X = df[df['Ground Truth'] == 4]
df_Y = df[df['Ground Truth'] == 5]

df_033 = df_sqr[df_sqr['size'] == 0.328125]
df_033['S'] = df_033['a'] *df_033['b']
#%% Calculate the difference b/w square and circle:
E = np.zeros((6, 512, 512))
E[0,:,:] = load_and_norm('dp_sim5-2-0.mat')    ## triangle
E[1,:,:] = load_and_norm('dp_sim25-4-5.mat')   ## rectangle
E[2,:,:] = load_and_norm('dp_sim0-5-3.mat')    ## circle
E[3,:,:] = load_and_norm('dp_sim15-2-2.mat')   ## zigzag
E[4,:,:] = load_and_norm('dp_sim5-5-4.mat')    ## +
E[5,:,:] = load_and_norm('dp_sim19-2-2.mat')   ## Y

R = np.zeros((6,6))  ## correlation matrix

for i in range(6):
    for j in range(6):
        R[i,j] = eval_Rxy(E[i,:,:], E[j,:,:])
        
plt.figure(figsize = [7.8, 6])
plt.pcolormesh(R, cmap = 'gray')
plt.colorbar()

EE = E[0,:,:]
EE1 = E[5,:,:]

plot(EE1)
plot(np.abs(EE1 - EE))
