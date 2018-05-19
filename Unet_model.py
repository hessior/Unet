import tensorflow as tf
from Unet_layers import *
import tensorflow.contrib.slim as slim

def deconv2d(input_, output_dim, ks=3, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)

def Unet(inpu, is_training, name="Unet"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1_1 = conv_layer(inpu, "conv1_1", [3, 3, 1, 64],is_training)
        conv1_2 = conv_layer(conv1_1, "conv1_2", [3, 3, 64, 64],is_training)
        pool1, pool1_index, shape_1 = max_pool(conv1_2, 'pool1')

        conv2_1 = conv_layer(pool1, "conv2_1", [3,3,64,128],is_training)
        conv2_2 = conv_layer(conv2_1, "conv2_2", [3,3,128,128],is_training)
        pool2, pool2_index, shape_2 = max_pool(conv2_2, "pool2")

        conv3_1 = conv_layer(pool2, "conv3_1", [3,3,128,256], is_training)
        conv3_2 = conv_layer(conv3_1, "conv3_2", [3,3,256,256], is_training)
        conv3_3 = conv_layer(conv3_2, "conv3_3", [3,3,256,256], is_training)
        pool3, pool3_index, shape_3 = max_pool(conv3_3, "pool3")

        conv4_1 = conv_layer(pool3, "conv4_1", [3,3,256,512], is_training)
        conv4_2 = conv_layer(conv4_1, "conv4_2", [3,3,512,512], is_training)
        conv4_3 = conv_layer(conv4_2, "conv4_3", [3,3,512,512], is_training)
        pool4, pool4_index, shape_4 = max_pool(conv4_3, "pool4")

        deconv4_1 = up_sampling(pool4, pool4_index, shape_4, "unpool4")
        deconv4_1 = tf.concat([deconv4_1, conv4_1],3)
        deconv4_2 = conv_layer(deconv4_1,"deconv4_2", [3, 3, 1024, 512],is_training)
        deconv4_3 = conv_layer(deconv4_2,"deconv4_3", [3, 3, 512, 512],is_training)
        deconv4_4 = conv_layer(deconv4_3,"deconv4_4", [3, 3, 512, 256],is_training)

        deconv3_1 = up_sampling(deconv4_4, pool3_index, shape_3, "unpool3")
        deconv3_1 = tf.concat([deconv3_1, conv3_1], 3)
        deconv3_2 = conv_layer(deconv3_1,"deconv3_2", [3, 3, 512, 256],is_training)
        deconv3_3 = conv_layer(deconv3_2,"deconv3_3", [3, 3, 256, 256],is_training)
        deconv3_4 = conv_layer(deconv3_3,"deconv3_4", [3, 3, 256, 128],is_training)
        
        deconv2_1 = up_sampling(deconv3_4, pool2_index, shape_2, "unpool2")
        deconv2_1 = tf.concat([deconv2_1, conv2_1], 3)
        deconv2_2 = conv_layer(deconv2_1,"deconv2_2", [3, 3, 256, 128],is_training)
        deconv2_3 = conv_layer(deconv2_2,"deconv2_3", [3, 3, 128, 64],is_training)
        
        deconv1_1 = up_sampling(deconv2_3, pool1_index, shape_1, "unpool1")
        deconv1_1 = tf.concat([deconv1_1, conv1_1], 3)
        deconv1_2 = conv_layer(deconv1_1,"deconv1_2", [3, 3, 128, 64],is_training)
        deconv1_3 = conv_layer(deconv1_2,"deconv1_3", [3, 3, 64, 64],is_training)

        filt = tf.get_variable("filter", shape=[1,1,64,1],
                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(deconv1_3, filt, [1,1,1,1], padding="SAME")
     
        logits = conv
        pred = tf.tanh(logits)

    return pred

##fake_img = tf.random_normal(shape=[3,160,256,1])
##is_training = tf.placeholder(tf.bool)
##test = Segnet2(fake_img, 4, 'test', is_training)
