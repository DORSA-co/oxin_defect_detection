import tensorflow as tf
from tensorflow import keras
import tensorflow as tf



def mask_binary_weight( hist ):
    hist = tf.Variable(hist, dtype=tf.float32)
    weights = 1/hist
    
    def loss_func(y_true, y_pred):
        
        binary_crossentropy_loss1 = y_true*tf.math.log(y_pred + 1e-7)
        binary_crossentropy_loss0 = (1-y_true)*tf.math.log(1-y_pred + 1e-7)
        loss = tf.reduce_mean(binary_crossentropy_loss0)  +tf.reduce_mean(binary_crossentropy_loss1)
        loss = loss * -1
        #loss = tf.reduce_mean(loss, axis=-1)
        return loss
    return loss_func







if __name__=='__main__':

    import numpy as np
    a = np.random.rand(16,32,200,4)
    b = np.random.rand(16,32,200,4)
    a = tf.Variable(a, dtype=tf.float32)
    b = tf.Variable(b, dtype=tf.float32)

    loss_func = mask_binary_weight([0.1]*4)
    c = loss_func(a,b)
    pass