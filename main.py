from tensorflow.python.keras import utils
from model import *
from data import *
import cv2
from matplotlib import pyplot as plt
from deep_utils import callbacks
import tensorflow as tf

gpu = tf.config.list_physical_devices('GPU') 
cpu = tf.config.list_physical_devices('CPU') 
tf.config.experimental.set_memory_growth(gpu[0], True)


batch = 8
epochs = 60


data_gen_args = dict(rotation_range=15,
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

#trainGen = trainGenerator(batch,'data/steel_defects/train','image','label',data_gen_args,save_to_dir = None, target_size=(128,800) )
#testGen = trainGenerator(batch,'data/steel_defects/test','image','label',data_gen_args2,save_to_dir = None, target_size=(128,800) )

trainGen = trainGenerator(batch,'data/steel_defect_class/train','image','label',data_gen_args,save_to_dir = None, target_size=(128,800) )
testGen = trainGenerator(batch,'data/steel_defect_class/test','image','label',data_gen_args2,save_to_dir = None, target_size=(128,800) )

train_data_count = 5333
test_data_count = 1333

'''
for j in range(10):
    x,y = next(trainGen)
    x = x * 255
    y = y * 255
    x = x.astype(np.uint8)
    y = y.astype(np.uint8)
    for i in range(len(x)):
        img = x[i]
        msk = y[i]

        img = img[:,:,0]
        msk = msk[:,:,0]

        cv2.imshow('mask',msk)
        cv2.imshow('img', img)
        cv2.waitKey(0)
'''


model = renet_unet(input_size=(128,800,1))
#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
my_callback = callbacks.CustomCallback('ch.h5')

'''
model.load_weights('ch.h5')

model.fit(  trainGen,
            steps_per_epoch=int(train_data_count/batch) + 1,
            epochs=epochs,
            callbacks=[my_callback ],
            validation_data=testGen,
            validation_steps=test_data_count//batch + 1, 
            initial_epoch=0)


model = renet_unet(input_size=(128,800,1), lr=1e-4)
model.load_weights('ch.h5')
model.fit(  trainGen,
            steps_per_epoch=int(train_data_count/batch) + 1,
            epochs=epochs,
            callbacks=[my_callback ],
            validation_data=testGen,
            validation_steps=test_data_count//batch + 1, 
            initial_epoch=30)


model.save('resnet_unet.h5')
'''
model.load_weights('resnet_unet.h5')


#model.save('steel_simple_unet.h5')
#model.load_weights('steel_simple_unet.h5')

#model.save('steel_unet.h5')
#model.load_weights('steel_unet.h5')



#model.load_weights('steel_simple_unet.h5')

path = 'data/steel_defects/test'
import cv2
import numpy as np
for fname in os.listdir( os.path.join(path, 'image')):
    
    lbl = cv2.imread(os.path.join( path, 'label/'+fname ),0)
    lbl = cv2.resize(lbl, (800,128))

    img = cv2.imread(os.path.join( path, 'image/'+fname ),0)
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
    cv2.waitKey(0)

#testGene = testGenerator("data/membrane/test")
#results = model.predict_generator(testGene,30,verbose=1)
#saveResult("data/membrane/test",results)