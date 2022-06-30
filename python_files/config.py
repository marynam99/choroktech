# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:10:54 2019

@author: liam.bui

The file contains configuration and shared variables

"""

SEED = 519

##########################
## FOLDER STURCTURE ######
##########################
WORK_DIRECTORY = r'C:\Users\SM-PC\PycharmProjects\choroktech'
CLT_WORK_DIRECTORY = r'C:\Users\SM-PC\HCILab\C:\Users\SM-PC\HCILab\chorok_dataset\안질환_crop중'
ORIGINAL_DATA_FOLDER = r'C:\Users\SM-PC\HCILab\dataset_cropped\train\before_split - 0511'
# CAT_DATASET_FOLDER = 'dataset'
CAT_DATASET_FOLDER = 'dataset_eye_train70'
TRAIN_FOLDER = 'train'
VAL_FOLDER = 'val'
TEST_FOLDER = 'test'
SAVE_IMG = 'result_images'

##########################
## EVALUATION METRICS ####
##########################
METRIC_ACCURACY = 'val_acc'
METRIC_F1_SCORE = 'f1-score'
METRIC_COHEN_KAPPA = 'Cohen kappa'
METRIC_CONFUSION_MATRIX = 'Confusion Matrix'
METRIC_TOP_K = 'top-k-acc'

###############
## MODEL ######
###############
# CAT_CLASSES = ['folliculitis', 'impetigo', 'normal', 'ringworm']
BIN_CLASSES = ['folliculitis', 'ringworm']

ALL_CAT = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '4', '5', '6', '7', '8', '9']
