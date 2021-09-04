from keras.datasets import cifar10
from keras.models import load_model

import numpy as np
import keras.utils.np_utils
import keras.utils as utils

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(_, _), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype('float32') / 255.0
y_test = utils.np_utils.to_categorical(y_test)

model = load_model(filepath='Image_Classifier.h5')

results = model.evaluate(x=x_test, y=y_test)

# print("Test loss:", results[0])
# print("Test accuracy:", results[1])

test_image_data = np.asarray([x_test[0]])

prediction_results = model.predict(x=test_image_data)
# Inputs into predict(x=...) has to be numpy arrays of images

max_index = np.argmax(prediction_results[0])

print("Prediction:", labels[max_index])
