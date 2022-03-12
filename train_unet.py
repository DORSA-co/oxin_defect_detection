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
train_path = 'data/mask/train'
test_path =  'data/mask/test'
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


trainGen = dataGenerator.maskGenerator(train_path, 'image','label', train_gen_args, target_size=input_size[:2], batch_size=8)
testGen = dataGenerator.maskGenerator (test_path , 'image','label', train_gen_args, target_size=input_size[:2], batch_size=8)



model = models.resnet_unet(input_size, num_class=1, mode=models.BINARY)
my_callback = callbacks.CustomCallback('checkpoint.h5')


#model.load_weights('ch.h5')

# model.fit(  trainGen,
#             steps_per_epoch=int(train_data_count/batch) + 1,
#             epochs=epochs,
#             callbacks=[my_callback ],
#             validation_data=testGen,
#             validation_steps=test_data_count//batch + 1, 
#             initial_epoch=0)

# model.save('resnet_unet.h5')
model.load_weights('resnet_unet.h5')



#_______________________________________________________________________________________________________________
#
#_______________________________________________________________________________________________________________
cnt=0
for fname in os.listdir( os.path.join(test_path, 'image')):
    
    lbl = cv2.imread(os.path.join( test_path, 'label/'+fname ),0)
    lbl = cv2.resize(lbl, (800,128))

    img = cv2.imread(os.path.join( test_path, 'image/'+fname ),0)
    img = cv2.resize(img, (800,128))

    inpt = np.expand_dims(img, axis=0)
    inpt = inpt.astype(np.float32) /255.
    out = model.predict(inpt)[0]

    ou1 = np.copy(out)
    thresh=0.3
    ou1[ou1>=thresh]= 1
    ou1[ou1<thresh] = 0
    ou1 = (ou1 * 255).astype(np.uint8)


    ou2 = np.copy(out)
    thresh=0.5
    ou2[ou2>=thresh]= 1
    ou2[ou2<thresh] = 0
    ou2 = (ou2 * 255).astype(np.uint8)
    ou2 = cv2.erode(ou2,np.ones((3,3)))
    ou2 = cv2.dilate(ou2,np.ones((3,3)))

    ou3 = np.copy(out)
    thresh=0.6
    ou3[ou3>=thresh]= 1
    ou3[ou3<thresh] = 0
    ou3 = (ou3 * 255).astype(np.uint8)


    cv2.imshow('img', img)
    cv2.imshow('lbl', cv2.bitwise_and(img,img, mask=lbl))
    cv2.imshow('out-0.3', cv2.bitwise_and(img,img, mask=ou1))
    cv2.imshow('out-0.5', cv2.bitwise_and(img,img, mask=ou2))
    cv2.imshow('out-0.7', cv2.bitwise_and(img,img, mask=ou3))
    key = cv2.waitKey(0)
    if key==ord('s') or key==ord('S'):
        cv2.imwrite('{}_img.jpg'.format(cnt),img)
        cv2.imwrite('{}_res.jpg'.format(cnt),cv2.bitwise_and(img,img, mask=ou2))
        cnt+=1
    
#_______________________________________________________________________________________________________________
#
#_______________________________________________________________________________________________________________