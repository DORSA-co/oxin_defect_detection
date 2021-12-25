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
epochs = 2
fine_tune_epochs=1
dataset_path = 'data/binary'
input_size = (128,800,3)


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    brightness_range=[0.6,1],
                    vertical_flip = True,
                    fill_mode='nearest',
                    )



trainGen, testGen = dataGenerator.get_binarygenerator( dataset_path,
                                                        (128,800),
                                                        'defective',
                                                        'perfect',
                                                        data_gen_args,
                                                        batch_size=batch,
                                                        validation_split=0.2)


model = models.xception_cnn(input_size, num_class=1, mode=models.BINARY)
my_callback = callbacks.CustomCallback('checkpoint_bin.h5')


inpt = input('do you want to train? y/n \n')
if inpt in ['Y','y']:
    model.fit(  trainGen,
                steps_per_epoch=trainGen.n//batch + 1,
                epochs=epochs-fine_tune_epochs,
                callbacks=[my_callback ],
                validation_data=testGen,
                validation_steps=testGen.n//batch + 1, 
                initial_epoch=0)

    model.save('binary_model.h5')
    #model.load_weights('binary_model.h5')



#_______________________________________________________________________________________________________________
#
#_______________________________________________________________________________________________________________



inpt = input('do you want to fine tune? y/n \n')
if inpt in ['Y','y']:
    
    model = models.xception_cnn(input_size, num_class=1, mode=models.CATEGORICAL, fine_tune_layer=120, weights='checkpoint_bin.h5')
    my_callback = callbacks.CustomCallback('checkpoint.h5')

    model.fit(  trainGen,
                steps_per_epoch=trainGen.n//batch + 1,
                epochs=epochs,
                callbacks=[my_callback ],
                validation_data=testGen,
                validation_steps=testGen.n//batch + 1, 
                initial_epoch=epochs-fine_tune_epochs)

    model.save('binary_model.h5')