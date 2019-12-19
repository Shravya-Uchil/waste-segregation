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
Run this code from the dataset-resized\dataset-resized folder or by using the augmented datset mentioned above. SVM model was trained by tuning various paramenters like the descritors(SIFT and SURF from openCV). Trained models seperately using augmented data and the basic original data.
### Using SIFT

## CNN Implementation:
We have tried multiple approaches of CNN implementaion and built the best possible CNN classifier for image segregation. In order to replicate our results and finding of the project run and execute the below files:
1. waste_classifier_CNN.py
2. wasteSegregation_CNN.py

You can also find the plots for different augmented and non augmented data in the repository.
We have also implemented VGG16 pre trained model in order to achieve good accuracy since the data set is small it did not perform well despite providing it with the trained weights of object detection.


