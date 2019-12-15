# Intelligent waste segregation using Machine Learning.

## Dataset: 
TrashNet data by Gary Thung - https://github.com/garythung/trashnet
Data consists of images belonging to six different categories:
Cardboard, Glass, Metal, Paper, Plastic, Trash.

## Data Augmentation procedure:
Refer - data_generator.py  
Run this code from dataset-resized\dataset-resized folder, i.e., the folder containing cardboard, glass, metal, etc ... folders.
The code will generate augmented images in gray scale. Images will be generated in cardboard_aug, glass_aug, ... folders.
Modify loopsize to generate more data if needed.
This needs to be run only once on each machine.
Note: Code is still crude and needs to be optimized.

## SVM Implemetation:
### Using SIFT

## CNN Implementation:
We have tried multiple approaches of CNN implementaion.  
waste_classifier_CNN.py

