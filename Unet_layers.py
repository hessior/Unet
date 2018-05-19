import tensorflow as tf
import numpy as np
import math


def up_sampling(inpu, ind, output_shape, name="up_sample"):
    with tf.variable_scope(name):
        batch_size = tf.cast(tf.shape(inpu),tf.int64)[0]
        # extend 
        inpu_ = tf.reshape(inpu, [-1])
        ind_ = tf.reshape(ind, [-1,1])
        
        batch_range = tf.reshape(tf.range(batch_size), [-1,1,1,1])
        b = tf.ones_like(ind) * batch_range     # same shape as inpu
        b = tf.reshape(b, [-1,1])               # indicate different image
        
        ind_ = tf.concat([b,ind_],1)
        ret = tf.scatter_nd(ind_, inpu_, shape=[batch_size, output_shape[1]*\
                                                output_shape[2]*output_shape[3]])
        ret = tf.reshape(ret, [-1, output_shape[1], output_shape[2], output_shape[3]])
    return ret    

    
######### special for segnet
def max_pool(inputs, name):
    with tf.variable_scope(name) as scope:
        value, index = tf.nn.max_pool_with_argmax(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                  padding='SAME', name=scope.name)
    return value, index, inputs.get_shape().as_list()
    # here value is the max value, index is the corresponding index, the detail information is here
    # https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/nn/max_pool_with_argmax


def conv_layer(inpu, name, shape, is_training):
    with tf.variable_scope(name):
        filt = tf.get_variable('filter',shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(inpu, filt, [1, 1, 1, 1], padding='SAME')
        batchnorm = tf.contrib.layers.batch_norm(conv, scale=True,is_training\
                                                 =is_training)
        conv_out = tf.nn.relu(batchnorm)
    return conv_out

