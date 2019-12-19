#!/usr/bin/env python
# coding: utf-8


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


surf = cv2.xfeatures2d.SURF_create()
images_path = "/home/014352130/ml_project/CNN_zip/data"

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
    images[i] = cv2.imread(files[i])
    images[i] = cv2.normalize(images[i], None, 0, 255, cv2.NORM_MINMAX)
    (kps, descs) = surf.detectAndCompute(images[i], None)
    
from sklearn.model_selection import train_test_split
images_train, images_test, y_train, y_test = train_test_split(images,labels, test_size=0.2, random_state=22)



def to_surf_desc(image):
    (kps, descs) = surf.detectAndCompute(np.array(image), None)
    #print (type(descs))
    #print (descs.shape)
    if descs is None:
        return np.empty((1,64))
    return descs
    
image_surf_train = [to_surf_desc(image) for image in images_train]
image_surf_test = [to_surf_desc(image) for image in images_test]


'''
print (len(image_surf_train))
for each in image_surf_train:
    print (each)'''
#print (image_surf_train[1][0])


# In[84]:



def BOVW(feature_descriptors, n_clusters = 64):
    print("Bag of visual words with {} clusters".format(n_clusters))
    #combined_features = np.vstack(np.array(feature_descriptors))
    combined_features = [point for image_keypoints in feature_descriptors for point in image_keypoints]
    print("Starting K-means training")
    kmeans_obj = KMeans(n_clusters=64, random_state=5).fit(combined_features)
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




model = svm.SVC(max_iter=10000, C=1000, kernel= 'poly', gamma=0.5, probability=True)
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
print(metrics.classification_report(y_test, y_test_predict, digits=6))
from  sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

#CVSVMModel = crossval(model,'Holdout',0.15)
scores = cross_val_score(model, x_train, y_train, cv=5)
print(scores)
'''
train_scores, valid_scores = validation_curve(model, x_train, y_train, "alpha", np.logspace(-7, 3, 3), cv=5)
plt.plot(train_scores, valid_scores)
plt.show()
$
'''
objects = ('training accuracy', 'test accuracy')
y_pos = np.arange(len(objects))
performance = [train_accuracy*100,test_accuracy*100]
plt.figure(1, figsize=(7, 5))
plt.bar(y_pos, performance, color = 'b', align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy Percentage')
plt.title('Accuracy plot 2500 samples - SURF descriptors and polynomial kernel')
plt.savefig('accuracy_SURF_polynomial.png')

import seaborn as sn
import pandas as pd
cm = pd.DataFrame(test_confusion_matrix, range(6), range(6))

sn.set(font_scale=1.4)#for label size
sn.heatmap(cm, annot=True,annot_kws={"size": 16})# font size
plt.savefig("confusion.png")

