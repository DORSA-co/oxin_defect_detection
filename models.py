import numpy as np 
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from tensorflow.keras import backend as keras
from deep_utils import metrics
from tensorflow.keras import layers 
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

BINARY = 'binary'
CATEGORICAL = 'categorical'

__loss__ = {CATEGORICAL:'categorical_crossentropy', BINARY:'binary_crossentropy'}
__activation__ = {CATEGORICAL:'softmax', BINARY:'sigmoid'}

#_____________________________________________________________________________________________________________________________
#
#_____________________________________________________________________________________________________________________________
def base_unet(input_size, num_class=1, lr=1e-3, mode=BINARY, pretained='imagenet', base_model='resnet34'):
    if input_size[-1]!=3:
        pretained = None
    model = sm.Unet(base_model, input_shape=input_size, classes=num_class, activation=__activation__[mode], encoder_weights=pretained)

    iou = metrics.iou()
    model.compile(optimizer = Adam(learning_rate=lr),
                  loss = __loss__[mode],
                  metrics = ['accuracy', tf.keras.metrics.Precision(name='Precision'),tf.keras.metrics.Recall(name='Recall'), iou])
    
    model.summary()
    return model


#_____________________________________________________________________________________________________________________________
#
#_____________________________________________________________________________________________________________________________
def unet(input_size , lr=1e-4, num_class=1, mode=BINARY, nfilter=64):
    assert num_class>0, 'num class could not be 0'
    if mode==CATEGORICAL:
        assert num_class>1, "for categorical out_mode, num_class should be greater than 2"



    inputs = Input(input_size)
    conv1 = Conv2D(nfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(nfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(nfilter*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(nfilter*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(nfilter*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(nfilter*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(nfilter*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(nfilter*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(nfilter*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(nfilter*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(nfilter*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(nfilter*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(nfilter*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(nfilter*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(nfilter*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(nfilter*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(nfilter*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(nfilter*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(nfilter*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(nfilter, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(nfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(nfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(num_class, 1, activation = __activation__[mode] )(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    iou = metrics.iou()
    '''
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                                initial_learning_rate,
                                                                decay_steps=1000,
                                                                decay_rate=0.95,
                                                                staircase=True
                                                                )
    '''
    model.compile(optimizer = Adam(lr =lr),
                  loss = __loss__[mode],
                  metrics = ['accuracy', tf.keras.metrics.Precision(name='Precision'),tf.keras.metrics.Recall(name='Recall'), iou])
    
    model.summary()


    return model

#_____________________________________________________________________________________________________________________________
#
#_____________________________________________________________________________________________________________________________
def low_unet(input_size ,lr=1e-4, num_class=1, mode=BINARY):
    assert num_class>0, 'num class could not be 0'
    if mode==CATEGORICAL:
        assert num_class>1, "for categorical out_mode, num_class should be greater than 2"
    
    inputs = Input(input_size)
    conv1 = SeparableConv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = SeparableConv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = SeparableConv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = SeparableConv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = SeparableConv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = SeparableConv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = SeparableConv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = SeparableConv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = SeparableConv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = SeparableConv2D(num_class, 1, activation = __activation__[mode] )(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    iou = metrics.iou()

    model.compile(optimizer = Adam(lr = lr),
                  loss = __loss__[mode], 
                  metrics = [
                      'accuracy',
                      tf.keras.metrics.Precision(name='Precision'),
                      tf.keras.metrics.Recall(name='Recall'), 
                      iou
                      ]
                  )  
    model.summary()

    return model






#_____________________________________________________________________________________________________________________________
#
#_____________________________________________________________________________________________________________________________
def resnet_unet(input_size,lr=1e-3, num_class=1, mode=BINARY):
    assert num_class>0, 'num class could not be 0'
    if mode==CATEGORICAL:
        assert num_class>1, "for categorical out_mode, num_class should be greater than 2"
    
    inputs = Input(shape=input_size)
    ### [First half of the network: downsampling inputs] ###
    # Entry block
    x =  Conv2D(32, 3, strides=2, padding="same")(inputs)
    x =  BatchNormalization()(x)
    x =  Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x =  Activation("relu")(x)
        x =  SeparableConv2D(filters, 3, padding="same")(x)
        x =  BatchNormalization()(x)

        x =  Activation("relu")(x)
        x =  SeparableConv2D(filters, 3, padding="same")(x)
        x =  BatchNormalization()(x)

        x =  MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual =  Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x =  add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x =  Activation("relu")(x)
        x =  Conv2DTranspose(filters, 3, padding="same")(x)
        x =  BatchNormalization()(x)

        x =  Activation("relu")(x)
        x =  Conv2DTranspose(filters, 3, padding="same")(x)
        x =  BatchNormalization()(x)

        x =  UpSampling2D(2)(x)

        # Project residual
        residual =  UpSampling2D(2)(previous_block_activation)
        residual =  Conv2D(filters, 1, padding="same")(residual)
        x =  add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs =  Conv2D(num_class, 3, activation=__activation__[mode], padding="same")(x)

    # Define the model
    model = Model(inputs, outputs)

    iou = metrics.iou()
    model.compile(optimizer = Adam(lr = lr),
                  loss = __loss__[mode],
                  metrics = [
                      'accuracy', 
                       tf.keras.metrics.Precision(name='Precision'),
                       tf.keras.metrics.Recall(name='Recall'), iou
                       ]
                  )
    
    model.summary()

    return model





#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------

#_____________________________________________________________________________________________________________
#
#_____________________________________________________________________________________________________________
def resnet_cnn(input_size,lr=1e-3, num_class=1, mode=BINARY, fine_tune_layer=-1, weights=None):
    preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=input_size)
    base_model.trainable = False

    inpt = tf.keras.Input(shape=input_size) 
    inpt_pre = preprocess_input(inpt)
    
    #--------------------------------------------
    out_base = base_model(inpt_pre)
    #--------------------------------------------
    x = tf.keras.layers.GlobalAveragePooling2D()(out_base)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    #x = keras.layers.Dense(256, activation='relu')(x)
    out = x = tf.keras.layers.Dense(num_class, activation=__activation__[mode] )(x)
    model = tf.keras.Model(inpt, out)
    #--------------------------------------------
    if weights is not None:
        model.load_weights(weights)
    #--------------------------------------------
    if fine_tune_layer>0:
        base_model = model.layers[3]
        base_model.trainable = True
        for i in range(fine_tune_layer):
            base_model.layers[i].trainable = False
    #--------------------------------------------        
    

        
    model.compile(optimizer = Adam(lr = lr),
                  loss = __loss__[mode], metrics = ['accuracy',
                                                    tf.keras.metrics.Precision(name='Precision'),
                                                    tf.keras.metrics.Recall(name='Recall')
                                                    ]
                  )
    
    model.summary()
    return model


#_____________________________________________________________________________________________________________________________
#
#_____________________________________________________________________________________________________________________________
def xception_cnn(input_size,lr=1e-3, num_class=1, mode=BINARY, fine_tune_layer=-1, weights=None):
    preprocess_input = tf.keras.applications.xception.preprocess_input
    base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=input_size)
    base_model.trainable = False

    inpt = tf.keras.Input(shape=input_size) 
    inpt_pre = preprocess_input(inpt)
    #--------------------------------------------
    out_base = base_model(inpt_pre)
    #--------------------------------------------
    x = tf.keras.layers.GlobalAveragePooling2D()(out_base)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    out = tf.keras.layers.Dense(num_class, activation= __activation__[mode] )(x)
    model = tf.keras.Model(inpt, out)
    
    #--------------------------------------------
    if weights is not None:
        model.load_weights(weights)
    #--------------------------------------------
    if fine_tune_layer>0:
        base_model = model.layers[3]
        base_model.trainable = True
        for i in range(fine_tune_layer):
            base_model.layers[i].trainable = False
    #--------------------------------------------
    
    model.compile(optimizer = Adam(lr = lr),
                  loss = __loss__[mode],
                  metrics = ['accuracy',
                             tf.keras.metrics.Precision(name='Precision'),
                             tf.keras.metrics.Recall(name='Recall')
                             ]
                 )
    return model

#_____________________________________________________________________________________________________________________________
#
#_____________________________________________________________________________________________________________________________


if __name__=='__main__':
    model = resnet_cnn( (128,800,3), num_class=5, mode=BINARY, fine_tune_layer=100 )
    
    end = True