
import tensorflow as tf
import pathlib
import numpy as np
import time

tflite_models_dir = pathlib.Path("exported/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"mnist_model.tflite"


interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()


input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
    
for _ in range(10):
    t = time.time()
    test_image = np.random.rand(1,640,960,1).astype(np.float32)

    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    print( time.time()- t)
    