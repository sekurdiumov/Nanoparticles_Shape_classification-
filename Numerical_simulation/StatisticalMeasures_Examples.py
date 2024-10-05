   # -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:09:03 2021

@author: aikch

Load the Example Count Matrix and Output Statistical Measures
such as precision and recall


"""

#%% Load official librarry


import scipy.io
import matplotlib.pyplot as plt
import numpy as np

#%% Load Customized library


from StatsPN import StatsPN

#plt.style.use('dark_background') 
plt.style.use('default') 
#%% Load the Count Matrix

#pth_rst = '10particlsCountMatrix.mat'
#pth_rst = '11ClassMatrix.mat'
pth_rst = 'Confusion_matrix_rot.mat'
file_Object = open(pth_rst,'rb')
MatFile = scipy.io.loadmat(file_Object)
file_Object.close()

locals().update(MatFile)

C = C_rot

plt.figure(1)
plt.pcolormesh(C, cmap = 'gray')
plt.colorbar()

#%%  Count Precision Recall, FPV, FNV

#Instantiate the Statistics Class
#Stats = StatsPN(CountMatrix)
Stats = StatsPN(C)

#Calculate the TP,NP,TN,FN
Stats.CalcBasis()
#Calculate metrics such as
    #precision and recall
Stats.CalcMetrics()


#%% Print the results

print('Precision:', Stats.precision)

print('\n')
print('Average precision = ', np.mean(Stats.precision))
print('\n')

print('Recall: ', Stats.recall)

print('\n')
print('Average recall = ', np.mean(Stats.recall))
print('\n')

#Fall-Out, False Positive Rate
print('FPR: ', Stats.FPR)


#False Omission Rate
print('FOR:', Stats.FOR)

fname = 'recall1_rot.mat'
scipy.io.savemat(fname, {'ConfMat': C, 'Precision': Stats.precision, 'Recall': Stats.recall})
