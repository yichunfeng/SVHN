#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:31:42 2019

@author: yi-chun
"""

import tensorflow as tf
import SVHN_CNNmodel
from data_preprocess import data

EPOCHS=120

#### MODEL ####

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


x = tf.placeholder(tf.float32, shape = [None, 256, 256, 1], name='Input_Data')
y = tf.placeholder(tf.float32, shape = [None, 10], name='Input_Labels')
keep_prob = tf.placeholder(tf.float32)# uesd in dropout
x_image = tf.reshape(x, [-1, 256, 256, 1])

def model(data):

    ## conv1 layer ##
    W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    ## max pooling1 ##
    h_pool1 = max_pool_2x2(h_conv1)

    ## conv2 layer ##
    W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    ## max pooling2 ##
    h_pool2 = max_pool_2x2(h_conv2)

    ## fc layer ##
    W_fc1 = weight_variable([8*8*64, 256])
    b_fc1 = bias_variable([256])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    ## dropout ##
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## output layer ##
    W_fc2 = weight_variable([256, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return prediction



#### training process ####
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)   

sess = tf.Session()
"""
# OLD or NEW version of tensorflow
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(EPOCHS):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 5 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))

"""

