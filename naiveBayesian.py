"""
Image is in format (width, length) : that is for face there are 
0 columns and 70 rows.
"""
import naiveBayesianHelper as bh
from pathlib import Path

DIGIT_TRAIN_IMAGES = r'data/digitdata/trainingimages'
DIGIT_TRAIN_LABELS = r'data/digitdata/traininglabels'
DIGIT_TEST_IMAGES = r'data/digitdata/testimages'
DIGIT_TEST_LABELS = r'data/digitdata/testlabels'

FACE_TRAIN_IMAGES = r'data/facedata/facedatatrain'
FACE_TRAIN_LABELS = r'data/facedata/facedatatrainlabels'
FACE_TEST_IMAGES = r'data/facedata/facedatatest'
FACE_TEST_LABELS = r'data/facedata/facedatatestlabels'

DIMENSIONS_DIGIT = (28, 28)
DIMENSIONS_FACE = (60, 70)
FEATURE_DIGIT_TUPLE = (1, 1)
FEATURE_FACE_TUPLE = (5, 5)

path_dict = {
    'digitdata': 
        {
        'paths': [DIGIT_TRAIN_IMAGES, DIGIT_TRAIN_LABELS, DIGIT_TEST_IMAGES, DIGIT_TEST_LABELS],
        'featuretuple' : FEATURE_DIGIT_TUPLE,
        'dimensions' : DIMENSIONS_DIGIT
        },
        
    'facedata' : 
        {
         'paths': [FACE_TRAIN_IMAGES, FACE_TRAIN_LABELS, FACE_TEST_IMAGES, FACE_TEST_LABELS],
         'featuretuple' : FEATURE_FACE_TUPLE,
         'dimensions' : DIMENSIONS_FACE
        }
}

"""
true_labels takes label file and stores true labels, which will be used
later to compare with predicted labels. 
"""
currentlyWorkingDS = input("Bayesian N/W on which dataset?: ")
splitParameter = int(input("Partition(Testing)?: "))/100

if(currentlyWorkingDS.lower() not in path_dict.keys()):
    print("Key did not match to any dataset, check again")
else:
    mname = "bayes_likelihood_"+currentlyWorkingDS.lower()+".pkl"
    pname = "bayes_prior_"+currentlyWorkingDS.lower()+".pkl"
    path = r"C:/Users/Dell/projectcs520/models/"
    checkExistL = Path(path+mname)
    checkExistP = Path(path+pname)
    
    if(checkExistL.exists() and checkExistP.exists()):
        print("Model already exists")
    else:
        print("Initializing Training on ", currentlyWorkingDS)
        bh.do_training(path_dict[currentlyWorkingDS.lower()], mname,pname, path, splitParameter)
        print("Training has been done and model has been saved with name: ",mname)
    
    print("===========================")
    print("Initializing Testing")
    bh.do_testing(path+mname,path+pname,path_dict[currentlyWorkingDS.lower()], splitParameter)