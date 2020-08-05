#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:57:37 2019

@author: yi-chun
"""

import numpy as np
#import pickle
from data_preprocess import data

tempA=np.array([[1,2],[3,4],[5,6]])
tempB=np.array([[11,22],[33,44],[55,66]])
print('tempA:',tempA)
print('shape of tempA:',tempA.shape)
print('tempB:',tempB)
print('shape of tempb:',tempB.shape)

reA=np.ndarray.flatten(tempA)
print('reA:',reA)
rea=reA.reshape(2,3)
print('rea:',rea)


reB=np.ndarray.flatten(tempB,order='F')
print('reB:',reB)
reb=reB.reshape(2,3,order='C')
print('reb:',reb)

#with open("dataset.pkl",'rb') as f: 
     #dataset = pickle.load(f) 
     
dataset=data   
print('dataset type:',type(dataset))
