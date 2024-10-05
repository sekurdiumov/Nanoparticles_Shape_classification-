# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:38:17 2023

@author: sk1u19
"""
import numpy as np
import random
import scipy.io as scio

N_test = 117
N_train = 1169 - 117

Test = np.ones(N_test)
Train = np.zeros(N_train)

Test = np.concatenate((Test, Train), axis = 0)

Test = Test.tolist()
random.shuffle(Test)
Test = np.array(Test)

inds1 = np.where(Test == 1)
inds1 = inds1[0]

inds0 = np.where(Test == 0)
inds0 = inds0[0]

test_1 = Test[inds1]
test_0 = Test[inds0]

scio.savemat('Test_selection.mat', {'Test_inds' : Test})