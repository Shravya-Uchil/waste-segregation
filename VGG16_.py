import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

K.set_image_dim_ordering('tf')

from keras.utils import np_utils

PATH = os.getcwd()
data_path = PATH + '/data_aug'
data_dir_list = os.listdir(data_path)

img_rows = 512
img_cols = 384
num_channels = 3

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset - ' + '{}\n'.format(dataset))
    for img in img_list:
        ip_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        ip_img = cv2.cvtColor(ip_img, cv2.COLOR_BGR2GRAY)
        ip_img = cv2.resize(ip_img, (224, 224))

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

labels[0:1600] = 0
labels[1601:3200] = 1
labels[3201:4800] = 2
labels[4801:6400] = 3
labels[6401:8000] = 4
labels[8001:] = 5

names = ['glass_aug', 'cardboard_aug', 'trash_aug', 'plastic_aug', 'metal_aug', 'paper_aug']

label_categorical = np_utils.to_categorical(labels, num_classes)

x, y = shuffle(img_data, label_categorical, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=2)

input_shape = img_data[0].shape

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


if __name__ == "__main__":
    model = VGG_16('/home/014370954/waste_segregation/vgg16weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=["accuracy"])

    num_epoch = 20
    hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch,
                     verbose=1, validation_split=0.33)# validation_data=(X_test, y_test))

    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    xc = range(num_epoch)

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    plt.style.use(['classic'])
    plt.savefig('Loss_Plot_deep.png')
    # plt.show()

    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    plt.style.use(['classic'])
    plt.savefig('Accuracy_Plot_deep.png')
    # plt.show()

    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test Loss', score[0])
    print('Train Loss', score[0])

    test_image = X_test[0:1]
    print(test_image.shape)
    print(model.predict(test_image))
    print(model.predict_classes(test_image))
    print(y_test[0:1])

    new_img = cv2.imread(PATH + '/data_aug/paper_aug/_362_8814706.png')
    print(new_img.shape)
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.resize(new_img, (224, 224))
    new_img = np.asarray(new_img)
    new_img = new_img.astype('float32')
    new_img /= 255
    print(new_img.shape)

    # new_img = np.expand_dims(new_img, axis=3)
    new_img = np.expand_dims(new_img, axis=0)
    print(new_img.shape)

    print(model.predict(new_img))
    print(model.predict_classes(new_img))
