#coding=utf-8

import tensorflow as tf
import numpy as np

data_dict = np.load('drive/My Drive/vgg/vgg16.npy', encoding='latin1', allow_pickle=True).item()

def print_layer(t):
    print (t.op.name, ' ', t.get_shape().as_list(), '\n')

def conv(x, d_out, name, finetune=False, xavier=False):
    d_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        # Fine-tuning
        if finetune:
            '''
            kernel = tf.Variable(tf.constant(data_dict[name][0]), name="weights")
            bias = tf.Variable(tf.constant(data_dict[name][1]), name="bias")
            '''
            kernel = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
            print ("finetune")
        elif not xavier:
            kernel = tf.Variable(tf.truncated_normal([3, 3, d_in, d_out], stddev=0.1), name='weights')
            bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),
                                                trainable=True,
                                                name='bias')
            print ("truncated_normal")
        else:
            kernel = tf.get_variable(scope+'weights', shape=[3, 3, d_in, d_out],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),
                                                trainable=True,
                                                name='bias')
            print ("xavier")
        conv = tf.nn.conv2d(x, kernel,[1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(conv + bias, name=scope)
        print_layer(activation)
        return activation

def maxpool(x, name):
    activation = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name=name)
    print_layer(activation)
    return activation

def fc(x, n_out, name, finetune=False, xavier=False):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if finetune:
            '''
            weight = tf.Variable(tf.constant(data_dict[name][0]), name="weights")
            bias = tf.Variable(tf.constant(data_dict[name][1]), name="bias")
            '''
            weight = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
            print ("finetune")
        elif not xavier:
            weight = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01), name='weights')
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]),
                                                trainable=True,
                                                name='bias')
            print ("truncated_normal")
        else:
            weight = tf.get_variable(scope+'weights', shape=[n_in, n_out],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]),
                                                trainable=True,
                                                name='bias')
            print ("xavier")
        #all conected layer uses relu_layer function
        activation = tf.nn.relu_layer(x, weight, bias, name=name)
        print_layer(activation)
        return activation

def VGG16(images, _dropout, n_cls):

    conv1_1 = conv(images, 64, 'conv1_1', finetune=True)
    conv1_2 = conv(conv1_1, 64, 'conv1_2', finetune=True)
    pool1   = maxpool(conv1_2, 'pool1')

    conv2_1 = conv(pool1, 128, 'conv2_1', finetune=True)
    conv2_2 = conv(conv2_1, 128, 'conv2_2', finetune=True)
    pool2   = maxpool(conv2_2, 'pool2')

    conv3_1 = conv(pool2, 256, 'conv3_1', finetune=True)
    conv3_2 = conv(conv3_1, 256, 'conv3_2', finetune=True)
    conv3_3 = conv(conv3_2, 256, 'conv3_3', finetune=True)
    pool3   = maxpool(conv3_3, 'pool3')

    conv4_1 = conv(pool3, 512, 'conv4_1', finetune=True)
    conv4_2 = conv(conv4_1, 512, 'conv4_2', finetune=True)
    conv4_3 = conv(conv4_2, 512, 'conv4_3', finetune=True)
    pool4   = maxpool(conv4_3, 'pool4')

    conv5_1 = conv(pool4, 512, 'conv5_1', finetune=True)
    conv5_2 = conv(conv5_1, 512, 'conv5_2', finetune=True)
    conv5_3 = conv(conv5_2, 512, 'conv5_3', finetune=True)
    pool5   = maxpool(conv5_3, 'pool5')

    # full connection layer
    flatten  = tf.reshape(pool5, [-1, 7*7*512])
    fc6      = fc(flatten, 4096, 'fc6', xavier=True)
    dropout1 = tf.nn.dropout(fc6, _dropout)

    fc7      = fc(dropout1, 4096, 'fc7', xavier=True)
    dropout2 = tf.nn.dropout(fc7, _dropout)
    
    fc8      = fc(dropout2, 17, 'fc8', xavier=True)
    dropout3 = tf.nn.dropout(fc8, _dropout)
    
    fc9      = fc(dropout3, 5, 'fc9', xavier=True)

    return fc9
