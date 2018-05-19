import numpy as np
import tensorflow as tf
import nibabel
from scipy.misc import imsave, imread
from scipy.io import loadmat

import glob
import time
import os

#### local packages
import Unet_model as unet
from img_trans import data_aug

##############   18/03/26  it worked

trainPATH = r"H:\ZZprojects\segnet_data\train_T1.npy"
testPATH = r"H:\ZZprojects\segnet_data\test_cCT.npy"
IMGtoPATH = "./my_output-{}/"
CheckPointPATH = "./ckpt/my_model.ckpt"
SummaryPATH = "./summary_Unet.txt"
SummaryNpyPATH = "./summary"

EPOCH = 310
BATCHSIZE = 16
DATALENGTH = 500
LEARNING_RATE = 0.0002
HEIGHT = 168
WIDTH = 256
CHANNEL = 1
from_epoch = 0


def msqre(logits, labels):
    ''' logits: [bch_size, height, width, channel]
        labels: [bch_size, height, width, channel]'''
    loss = tf.reduce_mean((logits - labels)**2)
    return loss

def mabse(logits, labels):
    ''' logits: [bch_size, height, width, channel]
        labels: [bch_size, height, width, channel]'''
    loss = tf.reduce_mean(tf.abs(logits - labels))
    return loss

### check loss
##fake_img=tf.constant(np.reshape(np.arange(1,18,2),[1,3,3,1]),dtype=tf.float32)
##fake_label=tf.constant(np.reshape(np.arange(9),[1,3,3,1]),dtype=tf.float32)
##check_loss = msqre(fake_img, fake_label)
##print(tf.Session().run(check_loss))

def train():
    tf.set_random_seed(100)
    
    imgs = tf.placeholder(tf.float32,[BATCHSIZE,HEIGHT,WIDTH,CHANNEL],name='imgs')
    labels = tf.placeholder(tf.float32,[BATCHSIZE,HEIGHT,WIDTH,1],name='label')
    lr = tf.placeholder(tf.float32, name="learning_rate")
    is_training = tf.placeholder(tf.bool, name="is_training")
    
    logits = unet.Unet(imgs, is_training, "Unet")
    sqr_loss = msqre(logits, labels)
    abs_loss = mabse(logits, labels)

    ## weighted loss (more weight for bone)
    loss = tf.reduce_mean((labels+2)**2*tf.abs(logits - labels))
    
    optimizer = tf.train.AdamOptimizer(lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss)
        
    saver = tf.train.Saver()
    
    train_imgs = (np.load(trainPATH) + 1) / 2
    train_label = (np.load(testPATH) + 1) /2
    train_imgs = train_imgs[:, 44:-44, :, :]
    train_label = train_label[:, 44:-44, :, :]
    test_imgs = train_imgs[(DATALENGTH):,:,:,:] * 2 - 1
    test_label = train_label[(DATALENGTH):,:,:,:] * 2 - 1
    train_imgs = train_imgs[:(DATALENGTH),:,:,:] 
    train_label = train_label[:(DATALENGTH),:,:,:]
    train_imgs, train_label = data_aug(train_imgs, train_label)
    train_imgs = train_imgs * 2 - 1
    train_label = train_label * 2 - 1
    train_set_length = train_imgs.shape[0]
    LOOPS = train_set_length // BATCHSIZE
    print("check: ", train_imgs.shape, train_imgs.max(), train_label.max())    

    summary_np = np.zeros([EPOCH+from_epoch, 4])
    file = open(SummaryPATH, 'a')
    file.write("Epoch  absloss  sqrloss  absloss_f  sqrloss_f\n")
    
    with tf.Session() as sess:
        if from_epoch==0:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess,  CheckPointPATH+"-{}".format(from_epoch))
        for epoch in range(from_epoch, from_epoch + EPOCH):
            lrate = LEARNING_RATE if epoch < 40 else LEARNING_RATE*(EPOCH - epoch)/160
            for i in range(LOOPS):
                tic = time.time()
                img_bch = train_imgs[(i*BATCHSIZE):((i+1)*BATCHSIZE),:,:,:]
                label_bch = train_label[(i*BATCHSIZE):((i+1)*BATCHSIZE),:,:,:]
                
                _, cur_loss = sess.run([train_step, loss],
                                       feed_dict={imgs:img_bch,
                                                  is_training:True,
                                                  labels:label_bch,lr:lrate})
                
                print("Epoch {}: BATCH {} Time: {:.4f} Loss:{:.4f} ".format(
                    epoch,i,time.time()-tic, cur_loss))
             
            if epoch % 10 == 0:
                saver.save(sess, CheckPointPATH, global_step = epoch)
                if not os.path.exists(IMGtoPATH.format(epoch)):
                    os.mkdir(IMGtoPATH.format(epoch))
                for j in range(test_imgs.shape[0]//BATCHSIZE):
                    out = sess.run(logits, feed_dict={is_training:False,
                        imgs:test_imgs[j*BATCHSIZE:(j+1)*BATCHSIZE,:,:,:]})
                    out_f = sess.run(logits, feed_dict={is_training:True,
                        imgs:test_imgs[j*BATCHSIZE:(j+1)*BATCHSIZE,:,:,:]})
                    for i in range(BATCHSIZE):
                        concat_out = np.concatenate((out[i,:,:,0],out_f[i,:,:,0],
                                                     test_label[j*BATCHSIZE+i,:,:,0]),axis=0)
                        imsave(os.path.join(IMGtoPATH.format(epoch), \
                                        "out{}.jpg".format(j*BATCHSIZE+i)),concat_out)
            absloss = 0; sqrloss = 0; absloss_f = 0; sqrloss_f = 0
            for j in range(test_imgs.shape[0]//BATCHSIZE):
                
                test_img_bch = test_imgs[j*BATCHSIZE:(j+1)*BATCHSIZE,:,:,:]
                test_label_bch = test_label[j*BATCHSIZE:(j+1)*BATCHSIZE,:,:,:]
                
                _absloss, _sqrloss = sess.run([abs_loss, sqr_loss],
                                               feed_dict={is_training: True,
                                                          imgs: test_img_bch,
                                                          labels: test_label_bch})
                absloss  += _absloss
                sqrloss += _sqrloss
                
                _absloss_f, _sqrloss_f = sess.run([abs_loss, sqr_loss],
                                               feed_dict={is_training: False,
                                                          imgs: test_img_bch,
                                                          labels: test_label_bch})
                absloss_f  += _absloss_f
                sqrloss_f += _sqrloss_f                
            absloss = absloss / test_imgs.shape[0]
            sqrloss = sqrloss / test_imgs.shape[0]
            absloss_f = absloss_f / test_imgs.shape[0]
            sqrloss_f = sqrloss_f / test_imgs.shape[0]            
            summary_np[epoch,:] = [absloss, sqrloss, absloss_f, sqrloss_f]
            file.write("Epoch {}: {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                       epoch, absloss, sqrloss, absloss_f, sqrloss_f))
            file.flush()
        np.save(SummaryNpyPATH, summary_np)
        file.close()
        

def test(from_epch):
    '''
    generate results from a given checkpoint
    Output: numpy array
    '''
    imgs = tf.placeholder(tf.float32,[BATCHSIZE,168,256,1])
    labels = tf.placeholder(tf.float32, [BATCHSIZE,168,256,1])
    lr = tf.placeholder(tf.float32, name="learning_rate")
    is_training = tf.placeholder(tf.bool, name="is_training")    
    logits = unet.Unet(imgs, is_training, "segnet")
    saver = tf.train.Saver()
    
    train_imgs = np.load(trainPATH)
    train_label = np.load(testPATH)
    test_imgs = train_imgs[:,44:-44,:,:]
    test_label = train_label[:,44:-44,:,:]
    
    result = np.zeros([test_imgs.shape[0],HEIGHT,WIDTH,1])
    with tf.Session() as sess:
        saver.restore(sess, CheckPointPATH+"-{}".format(from_epch))
        if not os.path.exists(testIMGtoPATH.format(from_epch)):
            os.mkdir(testIMGtoPATH.format(from_epch))
        for j in range(test_imgs.shape[0]//BATCHSIZE):
            out = sess.run(logits,feed_dict={
                        imgs:test_imgs[j*BATCHSIZE:(j+1)*BATCHSIZE,:,:,:],
                        is_training: True})
            result[j*BATCHSIZE:(j+1)*BATCHSIZE,:,:,:] = out
    print(result.shape)
    np.save("Gen_T1", result[:,:,:,:])

#### Data IO takes some time, I generated .npy file, so need not use this
##def prepare_data(path,name,label=False):
##    ## there is some issue in order using glob.glob, so use sorted()
##    imglist = []
##    nii_names = sorted(glob.glob(os.path.join(path,name)))
##    for nii_name in nii_names:
##        img_slice = nibabel.load(nii_name).get_data()[6:166,:,:]
##        for j in range(img_slice.shape[2]):
##            if np.max(img_slice[:,:,j:(j+1)]) != np.min(img_slice[:,:,j:(j+1)]):
##                imglist.append(np.array(img_slice[:,:,j:(j+1)]))
##    if label==False:
##        return np.array(imglist,dtype=np.float32)
##    else:
##        return np.array(imglist,dtype=np.uint8)

if __name__ == "__main__":
    train()
    #test(from_epch=300)
