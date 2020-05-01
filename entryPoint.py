import perceptronHelper as ph
import miraHelper as mh
import naiveBayesianHelper as bh
from pathlib import Path
import matplotlib.pyplot as plt

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

modelNames = {
    'perceptron' : 'percep_',
    'bayesian' : ['bayes_likelihood_','bayes_prior_'],
    'mira' : 'mira_'
    }

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

algorithmName = input("Which algorithm?")
splitParameter = int(input("Partition(Testing)?: "))/100
currentlyWorkingDS = input("%s algorithm on which dataset?" % (algorithmName))

if(currentlyWorkingDS.lower() not in path_dict.keys()):
    print("Key did not match to any dataset, check again")
else:
    path = r'models/'
    if(algorithmName=='bayesian'):
        mname = modelNames[algorithmName][0] + currentlyWorkingDS.lower() + ".pkl"
        pname = modelNames[algorithmName][1] + currentlyWorkingDS.lower() + ".pkl"
        
        checkExistL = Path(path+mname)
        checkExistP = Path(path+pname)
    
        if(checkExistL.exists() and checkExistP.exists()):
            print("Models already exist")
        else:
            print("Initializing Training on ", currentlyWorkingDS)
            bh.do_training(path_dict[currentlyWorkingDS.lower()], mname,pname, path, splitParameter)
            print("Training has been done and model has been saved with names:%s and %s " % (mname, pname))

        print("===========================")
        print("Initializing Testing")
        bh.do_testing(path+mname,path+pname,path_dict[currentlyWorkingDS.lower()], splitParameter)

    else:
        mname = modelNames[algorithmName] + currentlyWorkingDS.lower() + ".pkl"
        checkExist = Path(path+mname)
        if(checkExist.exists()):
            print("Model already exists")
        else:
            print("Initializing Training on ", currentlyWorkingDS)
            if(algorithmName == 'perceptron'):
                ph.do_training(path_dict[currentlyWorkingDS.lower()], 170, mname,path, splitParameter)
            else:
                mh.do_training(path_dict[currentlyWorkingDS.lower()], 170, mname,path, splitParameter)
            print("Training has been done and model has been saved with name: ",mname)
        print("===========================")
        print("Initializing Testing")
        if(algorithmName == 'perceptron'):
            ph.do_testing(path+mname, path_dict[currentlyWorkingDS.lower()],splitParameter)
        else:
            mh.do_testing(path+mname, path_dict[currentlyWorkingDS.lower()],splitParameter)