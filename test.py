from tensorflow.python.keras import utils
from model import *
from data import *
import cv2
from matplotlib import pyplot as plt
from deep_utils import callbacks

gpu = tf.config.list_physical_devices('GPU') 
cpu = tf.config.list_physical_devices('CPU') 
tf.config.experimental.set_memory_growth(gpu[0], True)

#tf.config.experimental.set_visible_devices(cpu[0])
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch = 8
epochs = 30


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    
                    
                    )

data_gen_args2 = dict(rotation_range=0.0,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    shear_range=0.0,
                    zoom_range=0.0,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    
                    
                    )

#myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
trainGen = trainGenerator(batch,'data/steel_defects/train','image','label',data_gen_args,save_to_dir = None, target_size=(128,800) )
testGen = trainGenerator(batch,'data/steel_defects/test','image','label',data_gen_args2,save_to_dir = None, target_size=(128,800) )

train_data_count = 5333
test_data_count = 1333
'''
x,y = next(myGene)
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


model = unet(input_size=(128,800,1))


model.load_weights('ch.h5')
model.load_weights('steel.h5')


path = 'data/steel/train'
import cv2
import numpy as np
for fname in os.listdir( os.path.join(path, 'image')):
    
    lbl = cv2.imread(os.path.join( path, 'label/'+fname ),0)
    lbl = cv2.resize(lbl, (800,128))

    img = cv2.imread(os.path.join( path, 'image/'+fname ),0)
    inpt = cv2.resize(img, (800,128))

    inpt = np.expand_dims(inpt, axis=0)
    inpt = inpt.astype(np.float32) /255.
    out = model.predict(inpt)[0]
    out[out>=0.5]=1
    out[out<0.5] = 0
    out = (out * 255).astype(np.uint8)
    out = cv2.resize(out, (800,128))
    cv2.imshow('img', img)
    cv2.imshow('lbl', lbl)
    cv2.imshow('out', out)
    cv2.waitKey(0)

#testGene = testGenerator("data/membrane/test")
#results = model.predict_generator(testGene,30,verbose=1)
#saveResult("data/membrane/test",results)