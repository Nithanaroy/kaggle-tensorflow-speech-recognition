# Created for Tensorflow Speech Recognition Competition on Kaggle

Competition home page https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/

## Installation
- Install python 3.x, pip, virtualenv
- Create a folder for vitualenv
- `source src/bin/activate`
- `cd src`
- `pip3 install -r requirements.txt`

## Peparing the data for training
- Download train and test data from https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data
- Extract them
- Run `vectorize_train_data()` in src/utils.py passing the path to the train/ extracted
- Run `save_data()` in src/utills.py to persist the vectorized train data

### Preparing sample data
- Run `prepare_sample()` in src/utills.py which creates a small sample from the large vector data generated above 
- The sample is used for quickly iterating various models

## To Develop
- run the commands in start.sh in order to start the jupyter notebook server
- Then open src/src/audio_recognition_cnn.ipynb and follow the guide

## Use the trained model for inference
- We will soon publish our trained model after the competition submission timeline ends

## Interesting stats about the data set
- (16000, 1) is the size of the largest audio vector 
