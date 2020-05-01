import driverFile as df
import readingData as rd
import numpy as np
import pickle

def do_training(pathDict, totalEpoch, mname, path, splitParameter):
    numberOfFeatures = int(pathDict['dimensions'][1] / pathDict['featuretuple'][0]) * int(pathDict['dimensions'][0]/ pathDict['featuretuple'][0])
    
    true_labels = rd.load_label(pathDict['paths'][1])
    allLabels = len(list(set(true_labels)))
    total_images = len(true_labels)
    
    print("===========================")
    print("Reading data")
    dataWithLabel, labels = df.createDataWithLabel('training',pathDict['paths'][0], true_labels, total_images, pathDict['featuretuple'], pathDict['dimensions'][1], pathDict['dimensions'][0],splitParameter)
    print("===========================")
    print("Initializing training")
    
    weights = np.random.rand(numberOfFeatures, allLabels)
    for epoch in range(int(totalEpoch)):
        print("Epoch %d/%d" %(epoch+1,totalEpoch))
        error = 0
        for i in range(len(dataWithLabel)):
            eachDict = dataWithLabel[i]
            
            curr_label = int(eachDict['label'])
            curr_feature = eachDict['features']
            temp_numpy = np.zeros((numberOfFeatures, 1))
            
            for j in range(len(curr_feature)):
                temp_tuple = curr_feature[j]
                temp_numpy[j] = temp_tuple[0] + temp_tuple[1]
        
            dot_product = np.dot(weights.T, temp_numpy)
            predicted_label = np.argmax(dot_product)
            
            if(predicted_label != curr_label):
                error = error + 1
                weights[:,curr_label] = weights[:,curr_label] + temp_numpy[:,0]
                weights[:,predicted_label] = weights[:,predicted_label] - temp_numpy[:,0]
        accuracy = 100-((error/total_images)*100)
        print("Accuracy: ",accuracy)
        if (accuracy == 100.0):
            break
    with open(path+mname,'wb') as file:
        pickle.dump(weights, file)

def do_testing(mname, pathDict, splitParameter):
    numberOfFeatures = int(pathDict['dimensions'][1] / pathDict['featuretuple'][0]) * int(pathDict['dimensions'][0]/ pathDict['featuretuple'][0])
    learnedWeights = pickle.load(open(mname,"rb"))

    true_labels = rd.load_label(pathDict['paths'][3])
    total_images = len(true_labels)
    
    print("===========================")
    print("Reading data")
    dataWithoutLabel = df.createDataWithLabel('testing',pathDict['paths'][2], true_labels, total_images, pathDict['featuretuple'], pathDict['dimensions'][1], pathDict['dimensions'][0],splitParameter)
    print("===========================")
    print("Initializing testing")
    
    error_test = 0
    for i in range(len(dataWithoutLabel)):
    
        eachDict = dataWithoutLabel[i]
        
        curr_feature = eachDict['features']
        temp_numpy = np.zeros((numberOfFeatures, 1))
        
        for j in range(len(curr_feature)):
            temp_tuple = curr_feature[j]
            temp_numpy[j] = temp_tuple[0] + temp_tuple[1]
    
        dot_product_test = np.dot(learnedWeights.T, temp_numpy)
        predicted_label = np.argmax(dot_product_test)
        if(predicted_label != int(true_labels[i])):
            error_test = error_test+ 1
    
    print("Total error in testing data: ", error_test)
    print("Accuracy: ",100-((error_test/total_images)*100))
    return (100-((error_test/total_images)*100))