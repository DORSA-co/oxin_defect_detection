import tensorflow as tf
from deep_utils import metrics
import pathlib

path = 'checkpoint.h5'

model = tf.keras.models.load_model(path, custom_objects={'__iou__': metrics.iou})

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()


tflite_models_dir = pathlib.Path("exported/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)