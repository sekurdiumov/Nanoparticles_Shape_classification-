# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:20:19 2023

@author: sk1u19
"""

import numpy as np
import scipy.io as scio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors
from matplotlib import cm
from matplotlib.path import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable

import statistics
from statistics import mode

#plt.style.use('dark_background') 
plt.style.use('default') 
matplotlib.rcParams.update({'font.size' : 25})

#%% =========================== Procedures ====================================
## Load predictions:
def load_preds(fname):
    
    M1 = scio.loadmat(fname)

    numbers = M1['Numbers']
    GroundTruth = M1['GroundTruth']
    size = M1['size']
    a = M1['a']
    b = M1['b']

    Predictions = M1['Predictions']

    Preds = np.zeros(len(Predictions))

    for i in range(len(Preds)):
        Preds[i] = np.argmax(Predictions[i,:])

    numbers0 = numbers[0,:]
    GroundTruth0 = GroundTruth[0,:]
    size0 = size[0,:]
    a0 = a[0,:]
    b0 = b[0,:]

    Correct = np.zeros_like(Preds)
    for i in range(len(Preds)):
        if GroundTruth0[i] == Preds[i]:
            Correct[i] = 1

    diction = {'Numbers' : numbers0,
               'GroundTruth' : GroundTruth0,
               'Predictions' : Preds,
               'Correct' : Correct,
               'size' : size0,
               'a' : a0,
               'b' : b0}

    df = pd.DataFrame(diction)

    df.sort_values(by = ['Numbers'], inplace = True)
    
    return df

## Define most frequent prediction for each object:
def most_common(a):
    return(mode(a))

## Building a confusion matrix:
def ConfMat(GroundTruth, Predictions):
    
    CM = np.zeros((5,5))
    
    for i in range(len(GroundTruth)):
        
        ii = int(GroundTruth[i])
        jj = int(Predictions[i])
        CM[jj,ii] = CM[jj,ii] + 1
        
    return CM

## Plotting confusion matrix:
def plotCM(CM, cmap):
    
    fig, ax = plt.subplots(figsize = [8.7, 7])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    CM = np.flipud(CM)

    #matplotlib.rcParams.update({'font.size' : 16})

    im = ax.matshow(CM, cmap= cmap)
    fig.colorbar(im, cax=cax, orientation='vertical')

    #matplotlib.rcParams.update({'font.size' : 10})

    for i in range(len(CM)):
        for j in range(len(CM)):
            c = CM[j, i]
            
            if c >= 50:
                color = 'white'
            else:
                color = 'black'
            
            ax.text(i, j, str(round(c, 1)), 
                    va='center', 
                    ha='center', 
                    color = color)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
def plot_errors(df, cmap, ind):
    
    zigzag_marker = zigzagmarker()
    
    markers = ['^', 'o', zigzag_marker, '+', '1']
    
    fig, ax = plt.subplots()
    sz = df['size']
    ar = df['aspect_ratio_redef']
    pred = df['Predictions']
    #pred = pred + 1
    colors = ['r', 'b', 'g', 'k', 'm']
    
    scatter = ax.scatter(sz[pred == ind], 
                         ar[pred == ind], 
                         c=colors[ind],
                         alpha=0.4,
                         marker=markers[ind])
    
    for i in range(5):
        if i != ind:
            scatter = ax.scatter(sz[pred == i], 
                                 ar[pred == i], 
                                 c=colors[i],
                                 alpha=0.4,
                                 marker=markers[i])
        else:
            pass
        
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    plt.xlabel('size, in λ')
    plt.ylabel('aspect ratio')
    scatter.set_clim(vmin = 0, vmax = 4)
    plt.show()
    
def zigzagmarker():
    
    k = 35.0

    verts = [
       (-k, k),  # point 1
       (0., k),  # point 2
       (0., k-2.),
       (-k, k-2.),
       (0., k-2.),
       (0., 0.),
       (-2., 0.),
       (-2., k-2.),
       (-2., 0.), 
       (k, 0.),
       (k, 2.),
       (0., 2.),
       (0., 0.),
       (k, 0.),
       (k, -k),
       (k-2., -k),
       (k-2., 0.)
    ]
    
    codes = [
        Path.MOVETO, #begin drawing
        Path.LINETO, #straight line
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO, 
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO
    ]
    
    path = Path(verts, codes)
    
    return path

def plot_errors_redef(df, cmap, ind):
    
    zigzag_marker = zigzagmarker()
    
    markers = ['^', 'o', zigzag_marker, '+', '1']
    
    fig, ax = plt.subplots()
    sz = df['size']
    ar = df['aspect_ratio_redef']
    pred = df['Predictions']
    #pred = pred + 1
    colors = ['r', 'b', 'g', 'k', 'm']
    
    scatter = ax.scatter(sz[pred == ind], 
                         ar[pred == ind], 
                         c=colors[ind],
                         alpha=0.3,
                         marker=markers[ind])
    
    for i in range(5):
        if i != ind:
            scatter = ax.scatter(sz[pred == i], 
                                 ar[pred == i], 
                                 c=colors[i],
                                 alpha=0.3,
                                 marker=markers[i])
        else:
            pass
        
        
    #handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    plt.xlabel('size, in λ')
    plt.ylabel('aspect ratio')
    scatter.set_clim(vmin = 0, vmax = 1)
    plt.show()
    
def redefine_AR(GroundTruth, a, b):
    
    if GroundTruth == 0:
        AR = 2/(np.sqrt(3))*b/a
        
    elif GroundTruth == 1:
        
        #AR = a/b
        
        if a >= b:
            AR = a/b
        else:
            AR = b/a
    
    elif GroundTruth == 2:  
        
        if a >= b:
            AR = np.sqrt(2*a**2 + b**2)/a
        else:
            AR = np.sqrt(2*b**2 + a**2)/b
    
    elif GroundTruth == 3:
        
        #AR = a/b
        
        if a >= b:
            AR = a/b
        else:
            AR = b/a
        
    elif GroundTruth == 4:
        AR = np.sqrt(2/3) * (a + b/np.sqrt(2)) / b
    
    return AR

def gradbar(x, y):
    
    # create a normalizer
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    # choose a colormap
    cmap = 'gray_r'
    # map values to a colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(y)
    
    
    fig, ax = plt.subplots()
    bars = ax.bar(x, y)
    
    ax = bars[0].axes
    lim = ax.get_xlim()+ax.get_ylim()
    for bar, val in zip(bars, y):
        grad = np.atleast_2d(np.linspace(0,val,256)).T
        bar.set_zorder(1)
        bar.set_facecolor('none')
        x, y = bar.get_xy()
        #w, h = bar.get_width(), bar.get_height()
        w = 0.2
        h = bar.get_height()
        ax.imshow(np.flip(grad), extent=[x,x+w,y,y+h], aspect='auto', zorder=1,interpolation='nearest', cmap=cmap, norm=norm)
    ax.axis(lim)
    cb = fig.colorbar(mappable)


## ========================== Main: ===========================================
#%% Collective fool processing
df1 = load_preds('Results_reg_5classes_4x_run1_4824.mat')
df2 = load_preds('Results_reg_5classes_4x_run2_4824.mat')
df3 = load_preds('Results_reg_5classes_4x_run3_4824.mat')
df4 = load_preds('Results_reg_5classes_4x_run4_4824.mat')
df5 = load_preds('Results_reg_5classes_4x_run5_4824.mat')
df6 = load_preds('Results_reg_5classes_4x_run6_4824.mat')
df7 = load_preds('Results_reg_5classes_4x_run7_4824.mat')
df8 = load_preds('Results_reg_5classes_4x_run8_4824.mat')
df9 = load_preds('Results_reg_5classes_4x_run9_4824.mat')
df10 = load_preds('Results_reg_5classes_4x_run10_4824.mat')
df11 = load_preds('Results_reg_5classes_4x_run11_4824.mat')
df12 = load_preds('Results_reg_5classes_4x_run12_4824.mat')
df13 = load_preds('Results_reg_5classes_4x_run13_4824.mat')
df14 = load_preds('Results_reg_5classes_4x_run14_4824.mat')
df15 = load_preds('Results_reg_5classes_4x_run15_4824.mat')
df16 = load_preds('Results_reg_5classes_4x_run16_4824.mat')
df17 = load_preds('Results_reg_5classes_4x_run17_4824.mat')
df18 = load_preds('Results_reg_5classes_4x_run18_4824.mat')
df19 = load_preds('Results_reg_5classes_4x_run19_4824.mat')
df20 = load_preds('Results_reg_5classes_4x_run20_4824.mat')

# df1 = load_preds('Results_reg_5classes_2x_run1_1206.mat')
# df2 = load_preds('Results_reg_5classes_2x_run2_1206.mat')
# df3 = load_preds('Results_reg_5classes_2x_run3_1206.mat')
# df4 = load_preds('Results_reg_5classes_2x_run4_1206.mat')
# df5 = load_preds('Results_reg_5classes_2x_run5_1206.mat')
# df6 = load_preds('Results_reg_5classes_2x_run6_1206.mat')
# df7 = load_preds('Results_reg_5classes_2x_run7_1206.mat')
# df8 = load_preds('Results_reg_5classes_2x_run8_1206.mat')
# df9 = load_preds('Results_reg_5classes_2x_run9_1206.mat')
# df10 = load_preds('Results_reg_5classes_2x_run10_1206.mat')
# df11 = load_preds('Results_reg_5classes_2x_run11_1206.mat')
# df12 = load_preds('Results_reg_5classes_2x_run12_1206.mat')
# df13 = load_preds('Results_reg_5classes_2x_run13_1206.mat')
# df14 = load_preds('Results_reg_5classes_2x_run14_1206.mat')
# df15 = load_preds('Results_reg_5classes_2x_run15_1206.mat')
# df16 = load_preds('Results_reg_5classes_2x_run16_1206.mat')
# df17 = load_preds('Results_reg_5classes_2x_run17_1206.mat')
# df18 = load_preds('Results_reg_5classes_2x_run18_1206.mat')
# df19 = load_preds('Results_reg_5classes_2x_run19_1206.mat')
# df20 = load_preds('Results_reg_5classes_2x_run20_1206.mat')

Numbers = np.array(df1['Numbers'])
GroundTruth = np.array(df1['GroundTruth'])
size = np.array(df1['size'])
a = np.array(df1['a'])
b = np.array(df1['b'])

sizes = np.unique(size)

Preds1 = np.array(df1['Predictions'])
Preds2 = np.array(df2['Predictions'])
Preds3 = np.array(df3['Predictions'])
Preds4 = np.array(df4['Predictions'])
Preds5 = np.array(df5['Predictions'])
Preds6 = np.array(df6['Predictions'])
Preds7 = np.array(df7['Predictions'])
Preds8 = np.array(df8['Predictions'])
Preds9 = np.array(df9['Predictions'])
Preds10 = np.array(df10['Predictions'])
Preds11 = np.array(df11['Predictions'])
Preds12 = np.array(df12['Predictions'])
Preds13 = np.array(df13['Predictions'])
Preds14 = np.array(df14['Predictions'])
Preds15 = np.array(df15['Predictions'])
Preds16 = np.array(df16['Predictions'])
Preds17 = np.array(df17['Predictions'])
Preds18 = np.array(df18['Predictions'])
Preds19 = np.array(df19['Predictions'])
Preds20 = np.array(df20['Predictions'])

N_fools = 20
Preds_all = np.zeros((len(Preds1), N_fools))

Preds_all[:,0] = Preds1
Preds_all[:,1] = Preds2
Preds_all[:,2] = Preds3
Preds_all[:,3] = Preds4
Preds_all[:,4] = Preds5
Preds_all[:,5] = Preds6
Preds_all[:,6] = Preds7
Preds_all[:,7] = Preds8
Preds_all[:,8] = Preds9
Preds_all[:,9] = Preds10
Preds_all[:,10] = Preds11
Preds_all[:,11] = Preds12
Preds_all[:,12] = Preds13
Preds_all[:,13] = Preds14
Preds_all[:,14] = Preds15
Preds_all[:,15] = Preds16
Preds_all[:,16] = Preds17
Preds_all[:,17] = Preds18
Preds_all[:,18] = Preds19
Preds_all[:,19] = Preds20

Acc_fool = np.zeros((N_fools))
for i in range(len(Preds_all)):
    for j in range(N_fools):
        if Preds_all[i,j] == GroundTruth[i]:
            Acc_fool[j] = Acc_fool[j] + 1
            
Acc_fool = Acc_fool/len(Preds_all)
Acc_fool_mean = np.mean(Acc_fool)

Preds = np.zeros(len(Preds_all))
for i in range(len(Preds_all)):
    Preds[i] = most_common(Preds_all[i,:])
    
counter = 0
for i in range(len(Preds)):
    if GroundTruth[i] == Preds[i]:
        counter = counter + 1
        
Accuracy = counter/len(Preds)


NN = np.arange(N_fools)

plt.figure()
plt.bar(NN, Acc_fool*100)
plt.xlabel('NN number')
plt.ylabel('Accuracy, %')
plt.ylim(70, 90)
plt.axhline(Accuracy*100, color = 'k')
plt.axhline(Acc_fool_mean*100, color = 'g')

print('Accuracy = ', Accuracy)

Correct = np.zeros_like(Preds)
for i in range(len(Preds)):
    if GroundTruth[i] == Preds[i]:
        Correct[i] = 1

aspect_ratio = a/b

AR = np.zeros_like(Preds)
for i in range(len(Preds)):
    AR[i]= redefine_AR(GroundTruth[i], a[i], b[i])

diction = {'Numbers' : Numbers,
           'GroundTruth' : GroundTruth,
           'Predictions' : Preds,
           'Correct' : Correct,
           'size' : size,
           'a' : a,
           'b' : b,
           'aspect_ratio' : aspect_ratio,
           'aspect_ratio_redef' : AR}

df = pd.DataFrame(diction)

df.sort_values(by = ['Numbers'], inplace = True)

fac = 4

data_NP = np.zeros((240*fac, 3))

data_NP[:,0] = size
data_NP[:,1] = AR
data_NP[:,2] = Correct

scio.savemat('data_for_Nikitas.mat', {'data_size_AR_Correct' : data_NP})

fname = 'all_data_fixed_orientation_1x.mat'

scio.savemat(fname, {'GroundTruth' : GroundTruth,
                     'Predictions' : Preds,
                     'Correct' : Correct,
                     'size' : size,
                     'a' : a,
                     'b' : b,
                     'aspect_ratio' : aspect_ratio,
                     'aspect_ratio_redef' : AR})
#%% Building a confusion matrix for collective fool results:
CM = ConfMat(GroundTruth, Preds)

## Plotting confusion matrix:

cmap = 'gray_r' 
#cmap = 'Blues'   

#plotCM(CM, cmap)

CM1 = CM * 100/(48*fac)

plotCM(CM1, cmap)

scio.savemat('Confusion_matrix_exp.mat', {'C' : CM1})

classes = np.arange(5)
Right = np.zeros(5)
for i in range(5):
    Right[i] = CM1[i,i]

plt.figure()
plt.bar(classes, Right)
plt.xlabel('class')
plt.ylabel('Acc per class, %')
plt.ylim(0, 100)
#%% Accuracy vs. size:
df025 = df[df['size'] <= 0.25]
df035 = df[(df['size'] > 0.25) & (df['size'] <= 0.35)]
df045 = df[(df['size'] > 0.35) & (df['size'] <= 0.45)]
df055 = df[(df['size'] > 0.45) & (df['size'] <= 0.55)]

Acc_vs_size = np.zeros(4)
Acc_vs_size[0] = ((df025['Correct']==1).sum())/len(df025)
Acc_vs_size[1] = ((df035['Correct']==1).sum())/len(df035)
Acc_vs_size[2] = ((df045['Correct']==1).sum())/len(df045)
Acc_vs_size[3] = ((df055['Correct']==1).sum())/len(df055)

matplotlib.rcParams.update({'font.size' : 16})

x = ['0.2', '0.3', '0.4', '0.5']
gradbar(x, Acc_vs_size*100)

plt.xlabel('size, in λ')
plt.ylabel('Accuracy, %')

#%% Errors analysis for each shape class and size:
plt.figure()
hist_GT = df['GroundTruth'].hist()

df_correct = df[df['Correct'] == 1]
df_error = df[df['Correct'] == 0]

error_type = np.zeros(len(df_error)) + 9
df_error['error_type'] = error_type

df_error['error_type'][((df_error['GroundTruth'] == 1) & 
                        (df_error['Predictions'] == 0))] = 0

df_error['error_type'][((df_error['GroundTruth'] == 2) & 
                        (df_error['Predictions'] == 4))] = 1

df_error['error_type'][((df_error['GroundTruth'] == 4) & 
                        (df_error['Predictions'] == 2))] = 2

df_error['error_type'][((df_error['GroundTruth'] == 3) & 
                        (df_error['Predictions'] == 0))] = 3

df_error['error_type'][((df_error['GroundTruth'] == 0) & 
                        (df_error['Predictions'] == 1))] = 4

df_error['error_type'][((df_error['GroundTruth'] == 3) & 
                        (df_error['Predictions'] == 4))] = 5

df_error['error_type'][((df_error['GroundTruth'] == 3) & 
                        (df_error['Predictions'] == 1))] = 6

df_error['error_type'][((df_error['GroundTruth'] == 2) & 
                        (df_error['Predictions'] == 3))] = 7

df_error['error_type'][((df_error['GroundTruth'] == 0) & 
                        (df_error['Predictions'] == 3))] = 8

df00 = df[df['GroundTruth'] == 0]
df01 = df[df['GroundTruth'] == 1]
df02 = df[df['GroundTruth'] == 2]
df03 = df[df['GroundTruth'] == 3]
df04 = df[df['GroundTruth'] == 4]

df0_0 = df00
df0_1 = df01
df0_2 = df02
df0_3 = df03
df0_4 = df04

# df0_0 = df00.drop_duplicates()
# df0_1 = df01.drop_duplicates()
# df0_2 = df02.drop_duplicates()
# df0_3 = df03.drop_duplicates()
# df0_4 = df04.drop_duplicates()

df_error0 = df00[df00['Correct'] == 0]
df_error1 = df01[df01['Correct'] == 0]
df_error2 = df02[df02['Correct'] == 0]
df_error3 = df03[df03['Correct'] == 0]
df_error4 = df04[df04['Correct'] == 0]

# df_error_0 = df_error0.drop_duplicates()
# df_error_1 = df_error1.drop_duplicates()
# df_error_2 = df_error2.drop_duplicates()
# df_error_3 = df_error3.drop_duplicates()
# df_error_4 = df_error4.drop_duplicates()

df_error_0 = df_error0
df_error_1 = df_error1
df_error_2 = df_error2
df_error_3 = df_error3
df_error_4 = df_error4

## Custom colormap:
cvals = [0, 1, 2, 3, 4]
colors = ["red", "blue", "green", "black", "magenta"]
norm=plt.Normalize(0, 4)
tuples = list(zip(map(norm, cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

## Binary colormap:
cvals = [0, 1]
colors = ["red", "green"]
norm=plt.Normalize(0, 1)
tuples = list(zip(map(norm, cvals), colors))
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

cvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
colors = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta", "brown", "gray", "purple"]
norm=plt.Normalize(0, 9)
tuples = list(zip(map(norm, cvals), colors))
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

## Plot the error distribution vs. size and redefined aspect ratio:
#plot_errors_redef(df_correct, cmap1) 
#plot_errors_redef(df_error, cmap1) 

fig, ax = plt.subplots(figsize = [8,6])
sz = df_correct['size']
ar = df_correct['aspect_ratio_redef']
sz1 = df_error['size']
ar1 = df_error['aspect_ratio_redef']
errtype = df_error['error_type']
#correct = df['Correct']
#pred = pred + 1
ax.scatter(sz, ar, c = 'white')
im = ax.scatter(sz1, ar1, c = errtype, cmap = cmap2)
#fig.colorbar(im)
# legend = ax.legend(*scatter.legend_elements(), 
#                     loc = "upper left")
#handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
plt.xlabel('size, in λ')
plt.ylabel('aspect ratio')
#scatter.set_clim(vmin = 0, vmax = 1)
plt.show()

#%% Accuracy vs. redefined aspect ratio:
# df15 = df[(df['aspect_ratio_redef'] <= 1)]
# df20 = df[(df['aspect_ratio_redef'] > 1) & (df['aspect_ratio_redef'] <= 1.7)]
# df25 = df[(df['aspect_ratio_redef'] > 1.7) & (df['aspect_ratio_redef'] <= 2.3)]
# df30 = df[(df['aspect_ratio_redef'] > 2.3) & (df['aspect_ratio_redef'] <= 3.0)]
# df35 = df[df['aspect_ratio_redef'] > 3]
# #df35 = df[(df['aspect_ratio_redef'] > 3) & (df['aspect_ratio_redef'] <= 3.5)]
# # df40 = df[(df['aspect_ratio_redef'] > 3.5) & (df['aspect_ratio_redef'] <= 4.0)]
# # df45 = df[(df['aspect_ratio_redef'] > 4) & (df['aspect_ratio_redef'] <= 4.5)]

# Acc_vs_ar = np.zeros(5)
# Acc_vs_ar[0] = ((df15['Correct']==1).sum())/len(df15)
# Acc_vs_ar[1] = ((df20['Correct']==1).sum())/len(df20)
# Acc_vs_ar[2] = ((df25['Correct']==1).sum())/len(df25)
# Acc_vs_ar[3] = ((df30['Correct']==1).sum())/len(df30)
# Acc_vs_ar[4] = ((df35['Correct']==1).sum())/len(df35)
# #Acc_vs_size[5] = ((df40['Correct']==1).sum())/len(df40)
# #Acc_vs_size[6] = ((df45['Correct']==1).sum())/len(df45)

# matplotlib.rcParams.update({'font.size' : 16})

# ar1 = ['0.7', '1.4', '2.0', '2.6', '3.4']
# gradbar(ar1, Acc_vs_ar*100)

# plt.xlabel('aspect ratio')
# plt.ylabel('Accuracy, %')

#x = np.array([0.25, 0.35, 0.45, 0.55])
#x = np.arange(5)
#plt.figure()
#plt.hist()
#plt.bar(x, Acc_vs_size*100, color = 'w')
#plt.bar(x, Acc_vs_size*100)
#plt.xticks(x, ['1,5', '2.0', '2.5', '3.0', '3.5'])
#plt.xlabel('aspect ratio (redefined)')
#plt.ylabel('Accuracy, %')


## Plot the scatters - error vs size and aspect ratio:

plot_errors(df0_0, cmap, 0)  
plot_errors(df0_1, cmap, 1)  
plot_errors(df0_2, cmap, 2)  
plot_errors(df0_3, cmap, 3)  
plot_errors(df0_4, cmap, 4) 


## Histogram for accuracies vs. redefined AR:
df.sort_values(by = ['aspect_ratio_redef'], inplace = True)

counter = 0
N1 = 192

Acc_AR = np.zeros(5)
x = np.zeros(5)
for i in range(5):
    
    df_AR = df.iloc[counter : counter + N1]
    
    Acc_AR[i] = (df_AR['Correct'] == 1).sum()/N1
    
    x[i] = np.median(df_AR['aspect_ratio_redef'])
    
    counter = counter + N1

gradbar(x, Acc_AR*100)

plt.xlabel('aspect ratio')
plt.ylabel('Accuracy, %')

df.sort_values(by = ['size'], inplace = True)

counter = 0
N1 = 192

Acc_sz = np.zeros(5)
y = np.zeros(5)
for i in range(5):
    
    df_sz = df.iloc[counter : counter + N1]
    
    Acc_sz[i] = (df_sz['Correct'] == 1).sum()/N1
    
    y[i] = np.median(df_sz['size'])
    
    counter = counter + N1
    
gradbar(y, Acc_sz*100)

plt.xlabel('size, in λ')
plt.ylabel('Accuracy, %')