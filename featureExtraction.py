import numpy as np

"""
Feature of particular image of particular position & index starts from (0,0)
e.g.:
    Let's say matrices[5] (from driverFile.py)
    Extract feature from 2nd row and 3rd column ; i=1, j=2
    i being row number and j being column number
params:
    test = input image in Numpy array object
    i = row number
    j = column number
returns:
    Numpy array of the specific feature
"""

def particular_feature(test, i, j):
    feature = np.zeros((4, 4))
    temp_row = 0
    for row in range(4*i, 4*(i+1)):
        temp_col = 0
        for column in range(4*j, 4*(j+1)):
            feature[temp_row][temp_col] = test[row][column]
            temp_col = temp_col + 1
        temp_row = temp_row + 1
    return feature

"""
This method is the extension of the above one. It generates all the features
at once
params:
    test = input image in Numpy array object
returns:
    List of numpy array of all features.

"""
def all_features(test):
    list_Feature = []
    for i in range(0, 7):
        for j in range(0, 7):
            single_feature = particular_feature(test, i, j)
            list_Feature.append(single_feature)
    return list_Feature

"""
It calculates the black and gray pixels from specific feature
params:
    feature: specific feature as a numpy array object

"""

def calculation_feature(feature):
    black_pixel = 0
    gray_pixel = 0
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            if (feature[i][j] ==  1):
                gray_pixel = gray_pixel + 1
            elif (feature[i][j] == 2):
                black_pixel = black_pixel + 1
    return gray_pixel, black_pixel


"""
This method takes input image generates it all the feature and then 
calculates its gray and black pixel and returns the list of the features
to the user.
"""

def all_at_once(test):
    list_Feature = [] 
    for i in range(0, 7):
        for j in range(0, 7):
            single_feature = particular_feature(test, i, j)
            gp , bp = calculation_feature(single_feature)
            list_Feature.append((gp, bp))
    
    return list_Feature