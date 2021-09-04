from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.datasets import cifar10

import keras.utils.np_utils
import keras.utils as utils

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = utils.np_utils.to_categorical(y_train)
y_test = utils.np_utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3),
                 activation='relu', padding='same',
                 kernel_constraint=maxnorm(3)))
# Conv2D(padding=...), specifies what kind of shape or size of the output
# padding='same', ensures that the output image doesn't shrink due to
# convolution, but max-pooling can shrink
# kernel_constraint=maxnorm(3), output of kernel feature has maximum
# value of only 3/the specified value, it is to not give a big number
# to max-pooling to compute

model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(units=512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=30, batch_size=32)

model.save(filepath='Image_Classifier.h5')
