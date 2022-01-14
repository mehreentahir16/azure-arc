import keras as K
import sys
import numpy as np
import json
import tensorflow as tf
import logging
import os

labels_map = {
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot',
}


def predict(model: tf.keras.Model, x: np.ndarray) -> tf.Tensor:
    y_prime = model(x, training=False)
    probabilities = tf.nn.softmax(y_prime, axis=1)
    predicted_indices = tf.math.argmax(input=probabilities, axis=1)
    return predicted_indices

def init():
    global model
    print("Python version: " + str(sys.version) + ", keras version: " + K.__version__)
    print("Executing init() method...")
    if 'AZUREML_MODEL_DIR' in os.environ: 
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './outputs/model/model.h5') 
    else: 
        model_path = './outputs/model/model.h5' 

    model = K.models.load_model(model_path) 
    print("loaded model...")

def run(raw_data):
    print("Executing run(raw_data) method...")
    data = np.array(json.loads(raw_data)['data'])
    data = np.reshape(data, (1,28,28,1))
    predicted_index = predict(model, data).numpy()[0]
    predicted_name = labels_map[predicted_index]

    logging.info('Predicted name: %s', predicted_name)

    logging.info('Run completed')
    return predicted_name
