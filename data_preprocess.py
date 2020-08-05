#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 23:04:30 2019

@author: yi-chun
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import pickle as pickle

# load data from file
path = "/Users/yi-chun/Downloads/PyCode/SVHN/test/"
filenames=os.listdir(path)
filenames.sort(key=lambda x:int(x[:-4])) # In ascending order
data_number=len(filenames) # total number of images
#print('total number of data:',len(filenames))
#print(filenames)

Shape=[]
I=[]

data=np.ndarray(shape=(data_number,32,32),dtype='float32') 
I_resize=[]
i=0

for filename in os.listdir(path):
    img = Image.open(os.path.join(path,filename))
    img = img.convert('L') #turn to grayscale image
    I.append(img) # image list
    arr=np.asarray(img)
    Shape.append(arr.shape) #record shape of each image array
    
    
    IMG=img.resize((32,32),resample=Image.LANCZOS)#resize image
    I_resize.append(IMG) # rezize_image list
    img_ndarray=np.asarray(IMG)
    data[i,:,:]=img_ndarray
    i+=1
    #print('process:',i)

test_dataset=data.reshape([data_number,32,32,1])

#with open('dataset.pkl','wb') as f:
    #pickle.dump(data, f)
    
#np.save('dataset.npy',data)
#print('test_dataset:',test_dataset)
#print('shape of test_dataset:',test_dataset.shape)
#print('data type of test_dataset:',test_dataset.dtype)
#print('type of test_dataset:',type(test_dataset))
print('test_dataset preprocess finished.') 

#load_data=np.load('dataset.npy')
#print('load_data:',load_data)
#print('shape of load_data:',load_data.shape)
#print('data type of load_data:',load_data.dtype)
#print('type of load_data:',type(load_data))
#dataset=data.reshape(30,256,256)#order=c
   
    
 

"""
#print original images
plt.figure(num=1, figsize=(8, 5))
for i in range(data_number):
    plt.subplot(6,5,i+1)
    plt.imshow(I[i])
#print resize images
plt.figure(num=2, figsize=(8, 5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(I_resize[i])
"""



