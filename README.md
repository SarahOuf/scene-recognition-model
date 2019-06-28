# scene-recognition-model

A deep learning model which can recognize a scene of 10 different classes: 
* inside an airport 
* Bakery 
* Bedroom 
* Greenhouse 
* Gym 
* Kitchen 
* Operating room 
* Pool 
* Restaurant 
* Toystore

The model achieved an accuracy of 94.5% on the test data by finetuning an inception-resnet-v2 model. 
This was a competition on Kaggle which won first place: https://www.kaggle.com/c/fcis-cs-deeplearningcompetition/leaderboard, Team14.

### Installation Guide

The project was run on google colab but if you want to run on your own device you will need to
install the following:
* cv2
* numpy
* tensorflow
* matplotlib
* glob
* PIL
* tqdm

## Dataset

The dataset that is available on the competition was converted into tfrecord to be easier to use in training.
The training data was split into training and validation sets, which can be found [here](https://drive.google.com/open?id=13dmnblt0nQ66XvVzCdNkYk2ceJV_kAuc).
The testing data can be found [here](https://drive.google.com/open?id=1OxU6IAxt70EvXrTLOG5QlAKGQ8naEl8a).

The training & validation data should be placed in folder: train_data and the testing data should be placed in folder: test_data.

## Model

The model used to achieve the accuracy can be found [here](https://drive.google.com/open?id=1v2i5JLlrDp302osUkr-awBa54hhhfGyi).

The files of the model should be placed in folder: models.

## Runing the project

* To train your own model you should use the file: training_inception_resnet_v2.py
* To run the model on the validation set use the file: validation_file.py
* To run the model on the testing data use the file: testing_file.py
