import json
import tensorflow as tf


#Fashion MNIST Dataset CNN model development: https://github.com/zalandoresearch/fashion-mnist
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

# load the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#no. of classes
num_classes = 10

x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test,  num_classes)

with open('mydata.json', 'w') as f:
    json.dump({"data": x_test[10].tolist()}, f)