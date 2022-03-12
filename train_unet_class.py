from tensorflow.python.keras import utils
import models
import dataGenerator
from matplotlib import pyplot as plt
from deep_utils import callbacks
import tensorflow as tf
import numpy as np
import cv2
import os

gpu = tf.config.list_physical_devices('GPU') 
cpu = tf.config.list_physical_devices('CPU') 
tf.config.experimental.set_memory_growth(gpu[0], True)


batch = 8
epochs = 60
train_path = 'data/mask_class/train'
test_path =  'data/mask_class/test'
train_data_count = 5333
test_data_count = 1333
input_size = (128,800,1)


train_gen_args = dict(rotation_range=15,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.05,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='constant',            
                    )

data_gen_args2 = dict(rotation_range=0.0,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    shear_range=0.0,
                    zoom_range=0.0,
                    horizontal_flip=True,
                    fill_mode='constant',                                        
                    )


trainGen = dataGenerator.maskGenerator(train_path,
                                       'image',
                                       'label',
                                       train_gen_args, 
                                       subfolders_mask=['0','1','2','3','4'], 
                                       target_size=input_size[:2],
                                       batch_size=8)

testGen = dataGenerator.maskGenerator (test_path , 
                                       'image',
                                       'label',
                                       train_gen_args,
                                       subfolders_mask=['0','1','2','3','4'], 
                                       target_size=input_size[:2],
                                       batch_size=8)



model = models.resnet_unet(input_size, num_class=5, mode=models.BINARY)
my_callback = callbacks.CustomCallback('checkpoint.h5')


#model.load_weights('ch.h5')
inpt = input('do you want to train? y/n \n')
if inpt in ['Y','y']:
    model.fit(  trainGen,
                steps_per_epoch=int(train_data_count/batch) + 1,
                epochs=epochs,
                callbacks=[my_callback ],
                validation_data=testGen,
                validation_steps=test_data_count//batch + 1, 
                initial_epoch=0)

    model.save('resnet_unet.h5')

model.load_weights('resnet_unet.h5')



#_______________________________________________________________________________________________________________
#
#_______________________________________________________________________________________________________________
#from matplotlib import pyplot as plt

for imgs,labels in testGen:
    
    predicts = model.predict(imgs)
    for i in range(len(imgs)):
        img = imgs[i]
        masks_lbl = labels[i]
        masks_pred = predicts[i]

        masks_pred[masks_pred>0.5]=1
        masks_pred[masks_pred<0.5]=0

        img*=255
        masks_lbl*=255
        masks_pred*=255

        img=img.astype(np.uint8)
        masks_lbl=masks_lbl.astype(np.uint8)
        masks_pred=masks_pred.astype(np.uint8)

        nclass = masks_lbl.shape[-1]
        for i in range(nclass):
            msk_lbl = masks_lbl[:,:,i]
            msk_pred = masks_pred[:,:,i]

            res_lbl = cv2.bitwise_and(img,img, mask=msk_lbl)
            res_pred = cv2.bitwise_and(img,img, mask=msk_pred)

            plt.subplot(nclass+1, 2 ,i*2+3)
            plt.imshow(res_lbl, cmap='gray', vmin=0, vmax=255)

            plt.subplot(nclass+1, 2 ,i*2+4)
            plt.imshow(res_pred, cmap='gray', vmin=0, vmax=255)
            #cv2.imshow('lbl_{}'.format(i), cv2.bitwise_and(img,img, mask=msk_lbl))
            #cv2.imshow('pred_{}'.format(i), cv2.bitwise_and(img,img, mask=msk_pred))

        plt.subplot(nclass+1, 1 ,1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

        #plt.subplot(nclass+1, 2 ,i*2+4)
        plt.show()
        #cv2.waitKey(0)


#_______________________________________________________________________________________________________________
#
#_______________________________________________________________________________________________________________