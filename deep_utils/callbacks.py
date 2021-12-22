from tensorflow import keras
import os

class CustomCallback(keras.callbacks.Callback):
    
    def __init__(self,out_path):
        #super.__init__(self)
        self.out_path = out_path

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        self.model.save( self.out_path)
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))