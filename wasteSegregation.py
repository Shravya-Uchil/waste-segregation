from __future__ import print_function
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
import os
from keras.utils import np_utils
import cv2
import numpy as np
from sklearn.utils import shuffle

batch_size = 128
num_classes = 5
epochs = 10

PATH = os.getcwd()
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_data_list = []


img_x, img_y = 512, 384

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset - ' + '{}\n'.format(dataset))
    for img in img_list:
        ip_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        ip_img = cv2.cvtColor(ip_img, cv2.COLOR_BGR2GRAY)
        ip_img = cv2.resize(ip_img, (img_x, img_y))

        img_data_list.append(ip_img)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)

img_data = np.expand_dims(img_data, axis=4)
print(img_data.shape)

num_classes = 6

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

labels[0:402] = 0
labels[403:903] = 1
labels[904:1314] = 2
labels[1315:1908] = 3
labels[1909:2390] = 4
labels[2391:] = 5

names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

label_categorical = np_utils.to_categorical(labels, num_classes)

x, y = shuffle(img_data, label_categorical, random_state=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=2)

#x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
#x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 3)
input_shape = (img_y, img_x, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
