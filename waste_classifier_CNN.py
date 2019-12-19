import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix
import pandas as pd
import itertools
import seaborn as sn

K.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import regularizers

PATH = os.getcwd()
images_path = PATH + '/data_ML'
types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

files = []
labels = []

img_rows = 64
img_cols = 64
num_channels = 3
img_data_list = []
for i, each_type in enumerate(types):
    path = join(images_path, each_type) 
    for f in listdir(path):
        file_path = join(path, f)
        if isfile(file_path):
            files.append(file_path)
            labels.append(i)
images = np.empty(len(files), dtype=object)
for i in range(len(files)):
    ip_img = cv2.imread(files[i])
    ip_img = cv2.resize(ip_img, (img_rows, img_cols))
    img_data_list.append(ip_img)


        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)


num_classes = 6

label_categorical = np_utils.to_categorical(labels, num_classes)

x, y = shuffle(img_data, label_categorical, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=2)

input_shape = img_data[0].shape
print(input_shape)

model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
model.add(Convolution2D(8, 3, 3))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(8, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Dropout(0.5))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Dropout(0.5))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, kernel_regularizer=regularizers.l2(0.001)))#, activity_regularizer=regularizers.l1(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.8))

model.add(Dense(256, kernel_regularizer=regularizers.l2(0.001)))#, activity_regularizer=regularizers.l1(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adamax', metrics=["accuracy"])

num_epoch = 100
hist = model.fit(X_train, y_train, batch_size=8, nb_epoch=num_epoch,
                 verbose=1, validation_data=(X_test, y_test))

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = range(num_epoch)

#Confusion Matrix:
preds = np.round(model.predict(X_test))
categorical_test_labels = pd.DataFrame(y_test).idxmax(axis=1)
categorical_preds = pd.DataFrame(preds).idxmax(axis=1)
confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)
print('confusion_matrix',confusion_matrix)


def plot_confusion_matrix(cm, classes,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues):
 
#Add Normalization Option
#prints pretty confusion metric with normalization option        
    #cm = pd.DataFrame(test_confusion_matrix, range(6), range(6))
    
    sns.heatmap(confusion_matrix, cmap="BuPu", annot=True,cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion.png')

   
plot_confusion_matrix(confusion_matrix, ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'], normalize=False)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
plt.style.use(['classic'])
plt.savefig('Loss_Plot_100.png')
#plt.show()

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
plt.style.use(['classic'])
plt.savefig('Accuracy_Plot_100.png')
#plt.show()

score = model.evaluate(X_test, y_test, verbose=1)
print('Test Loss', score[0])
print('Train Loss', score[0])

test_image = X_test[0:1]
print('test_image X_test[0:1]: ', test_image)
print('test_image.shape: ', test_image.shape)
print('model.predict(test_image): ', model.predict(test_image))
print('model.predict_classes(test_image): ', model.predict_classes(test_image))
print('y_test[0:1]: ', y_test[0:1])

#Testing each image:

new_img = cv2.imread(images_path + '/trash/_10_5111875.png')
print('trash image test: ', new_img.shape)
#new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
new_img = cv2.resize(new_img, (64, 64))
new_img = np.asarray(new_img)
new_img = new_img.astype('float32')
new_img /= 255
print('trash.shape: ', new_img.shape)

#new_img = np.expand_dims(new_img, axis=3)
new_img = np.expand_dims(new_img, axis=0)
print('expand dims ', new_img.shape)

print('trash predict as: ',model.predict(new_img))
print('trash predict class as : ',model.predict_classes(new_img))


new_img = cv2.imread(images_path + '/plastic/_24_1575208.png')
print('plastic: ', new_img.shape)
#new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
new_img = cv2.resize(new_img, (64, 64))
new_img = np.asarray(new_img)
new_img = new_img.astype('float32')
new_img /= 255
#new_img = np.expand_dims(new_img, axis=3)
new_img = np.expand_dims(new_img, axis=0)
print('plastic predict as : ',model.predict(new_img))
print('plastic predict class as : ',model.predict_classes(new_img))


new_img = cv2.imread(images_path + '/cardboard/_43_2330194.png')
print('cardboard: ', new_img.shape)
#new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
new_img = cv2.resize(new_img, (64, 64))
new_img = np.asarray(new_img)
new_img = new_img.astype('float32')
new_img /= 255
new_img = np.expand_dims(new_img, axis=0)
print('cardboard predict as : ',model.predict(new_img))
print('cardboard predict class as : ',model.predict_classes(new_img))


new_img = cv2.imread(images_path + '/glass/_28_2221118.png')
print('glass: ', new_img.shape)
#new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
new_img = cv2.resize(new_img, (64, 64))
new_img = np.asarray(new_img)
new_img = new_img.astype('float32')
new_img /= 255
#new_img = np.expand_dims(new_img, axis=3)
new_img = np.expand_dims(new_img, axis=0)
print('glass predict: ',model.predict(new_img))
print('glass predict class: ',model.predict_classes(new_img))


new_img = cv2.imread(images_path + '/paper/_60_8721308.png')
print('paper: ', new_img.shape)
#new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
new_img = cv2.resize(new_img, (64, 64))
new_img = np.asarray(new_img)
new_img = new_img.astype('float32')
new_img /= 255
#new_img = np.expand_dims(new_img, axis=3)
new_img = np.expand_dims(new_img, axis=0)
print('paper predict: ',model.predict(new_img))
print('paper predict class: ',model.predict_classes(new_img))


new_img = cv2.imread(images_path + '/metal/_60_2657853.png')
print('metal: ', new_img.shape)
#new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
new_img = cv2.resize(new_img, (64, 64))
new_img = np.asarray(new_img)
new_img = new_img.astype('float32')
new_img /= 255
#new_img = np.expand_dims(new_img, axis=3)
new_img = np.expand_dims(new_img, axis=0)
print('metal predict: ',model.predict(new_img))
print('metal predict class: ',model.predict_classes(new_img))
#
