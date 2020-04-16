import probabilityCalculation as pc
import driverFile as df
import readingData as rd

DIGIT_TRAIN_IMAGES = r'C:/Users/Dell/projectcs520/data/digitdata/trainingimages'
DIGIT_TRAIN_LABELS = r'C:/Users/Dell/projectcs520/data/digitdata/traininglabels'
DIGIT_TEST_IMAGES = r'C:/Users/Dell/projectcs520/data/digitdata/testimages'
DIGIT_TEST_LABELS = r'C:/Users/Dell/projectcs520/data/digitdata/testlabels'

FACE_TRAIN_IMAGES = r'C:/Users/Dell/projectcs520/data/facedata/facedatatrain'
FACE_TRAIN_LABELS = r'C:/Users/Dell/projectcs520/data/facedata/facedatatrainlabels'
FACE_TEST_IMAGES = r'C:/Users/Dell/projectcs520/data/facedata/facedatatest'
FACE_TEST_LABELS = r'C:/Users/Dell/projectcs520/data/facedata/facedatatestlabels'

DIMENSIONS_DIGIT = (28, 28)
DIMENSIONS_FACE = (60, 70)
FEATURE_DIGIT_TUPLE = (4, 4)
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
def do_Training(currentlyWorkingDS):
    directory = path_dict[currentlyWorkingDS.lower()]

    true_labels = rd.load_label(directory['paths'][1])
    allLabels = len(list(set(true_labels)))
    total_images = len(true_labels)

    dataWithLabel = df.createDataWithLabel('training',directory['paths'][0], true_labels, total_images, directory['featuretuple'], directory['dimensions'][1], directory['dimensions'][0])
    trainingDict = pc.training_Bayesian(dataWithLabel,allLabels, directory['featuretuple'], directory['dimensions'][1], directory['dimensions'][0])

    prior_prob = {}
    for each_label in range(allLabels):
        val, tot = pc.calculatePrior(directory['paths'][1], each_label)
        prior_prob[each_label] = (val, tot)

    predicted_value = pc.posteriorProbability(dataWithLabel, allLabels, prior_prob, trainingDict)

    diff = 0
    for i in range(len(true_labels)):
        if(int(true_labels[i]) != predicted_value[i]):
            diff = diff + 1
    
    accuracy = 100-((diff*100)/total_images)
    
    return accuracy, trainingDict, prior_prob

def do_Testing(currentlyWorkingDS, prior_prob, trainingDict):
    directory = path_dict[currentlyWorkingDS.lower()]

    true_labels = rd.load_label(directory['paths'][3])
    allLabels = len(list(set(true_labels)))
    total_images = len(true_labels)

    processedData = df.createDataWithLabel('testing',directory['paths'][2], true_labels, total_images, directory['featuretuple'], directory['dimensions'][1], directory['dimensions'][0])
    predicted_value = pc.posteriorProbability(processedData, allLabels, prior_prob, trainingDict)

    diff = 0
    for i in range(len(true_labels)):
        if(int(true_labels[i]) != predicted_value[i]):
            diff = diff + 1
    
    accuracy = 100-((diff*100)/total_images)
    
    return accuracy
    
currentlyWorkingDS = input("Bayesian N/W on which dataset?")
if(currentlyWorkingDS.lower() not in path_dict.keys()):
    print("Key did not match to any dataset, check again")
else:
    print("Initializing Training on ", currentlyWorkingDS)
    
    accuracyTraining, trainingMat, prior_prob = do_Training(currentlyWorkingDS)
    
    print("Training done with accuracy ",accuracyTraining)
    print("Initializing Testing")
    
    accuracyTesting = do_Testing(currentlyWorkingDS, prior_prob, trainingMat)
    
    print("Accuracy in testing data is ",accuracyTesting)
