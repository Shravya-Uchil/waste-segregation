#!/usr/bin/env python
# coding: utf-8

# In[78]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cluster import KMeans
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
import csv
import keras
import keras.utils
from keras import utils as np_utils


# In[79]:


from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD


# In[80]:


'''
img2 = cv2.drawKeypoints(image1,kps,None,(255,0,0),4)
plt.imshow(img2),plt.show
'''


# In[81]:




images_path = '/home/navya/Documents/sjsu/project-257/dataset-resized/dataset-resized'

# CARDBOARD = 0 GLASS = 1 METAL = 2 PAPER = 3 PLASTIC = 4 TRASH = 5
types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

files = []
labels = []
for i, each_type in enumerate(types):
    path = join(images_path, each_type) 
    for f in listdir(path):
        file_path = join(path, f)
        if isfile(file_path):
            files.append(file_path)
            labels.append(i)

images = np.empty(len(files), dtype=object)
for i in range(len(files)):
    images[i] = cv2.imread(files[i], 0)
    images[i] = cv2.normalize(images[i], None, 0, 255, cv2.NORM_MINMAX)
    (kps, descs) = surf.detectAndCompute(images[i], None)
    
from sklearn.model_selection import train_test_split
images_train, images_test, y_train, y_test = train_test_split(images,labels, test_size=0.1, random_state=42)



# In[82]:


def to_surf_desc(image):
    (kps, descs) = surf.detectAndCompute(np.array(image), None)
    #print (type(descs))
    #print (descs.shape)
    if descs is None:
        return np.empty((1, 64))
    return descs
    
image_surf_train = [to_surf_desc(image) for image in images_train]
image_surf_test = [to_surf_desc(image) for image in images_test]

print (image_surf_train)
#print("kps : {}, descriptors : {}".format(len(kps), descs.shape))


# In[83]:


print (len(image_surf_train))
for each in image_surf_train:
    print (each)
#print (image_surf_train[1][0])


# In[84]:



def BOVW(feature_descriptors, n_clusters = 15):
    print("Bag of visual words with {} clusters".format(n_clusters))
    #combined_features = np.vstack(np.array(feature_descriptors))
    combined_features = [point for image_keypoints in feature_descriptors for point in image_keypoints]
    print("Starting K-means training")
    kmeans_obj = KMeans(n_clusters=15, random_state=0).fit(combined_features)
    return kmeans_obj

def predict_bow_features(kmeans, descs):
    
    X = []
    for (i, desc) in enumerate(descs):
        if len(desc) == 0:
            continue
        result = kmeans.predict(desc)
        cluster_vector = [(result == i).sum() for i in range(0, kmeans.n_clusters)]
        X.append(cluster_vector)
    return np.array(X)
    


# In[85]:



#from collections import Counter

kmeans = BOVW(image_surf_train)
x_train = predict_bow_features(kmeans, image_surf_train)
x_test = predict_bow_features(kmeans, image_surf_test)

'''
counter = Counter(result)
total=0
for val in counter.values():
    total += val
print(total)
plt.hist(result, bins=15)
'''


# In[86]:




model = svm.SVC(max_iter=10000, C=100, gamma=0.5, probability=True)
model.fit(x_train, y_train)


# In[87]:



confidence_of_model = model.predict_proba(x_train)
confidence_example = np.amax(confidence_of_model, axis=1)
print(stats.describe(confidence_example))


# In[93]:


def get_accuracy(y_predict, y):
    accuracy = np.sum(y_predict == y).__float__() / float(len(y))
    confusion_matrix = metrics.confusion_matrix(y_predict, y, labels=[0, 1, 2, 3, 4, 5])
    return (accuracy, confusion_matrix)


# In[94]:


# Test SVM model

y_train_predict = np.array(model.predict(x_train))
y_test_predict = np.array(model.predict(x_test))
(train_accuracy, train_confusion_matrix) = get_accuracy(y_train_predict, y_train)
(test_accuracy, test_confusion_matrix) = get_accuracy(y_test_predict, y_test)
print('Train accuracy: ' + str(train_accuracy))
print('Train confusion matrix:')
print(train_confusion_matrix)
print('Test accuracy: ' + str(test_accuracy))
print('Test confusion matrix:')
print(test_confusion_matrix)


# In[3]:


from sklearn.model_selection import cross_val_score
 
#CVSVMModel = crossval(model,'Holdout',0.15)
scores = cross_val_score(model, x_train, y_train, cv=5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


import cv2


# In[6]:


cv2.__version__


# In[ ]:



