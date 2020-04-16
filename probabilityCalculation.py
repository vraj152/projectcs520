import math
"""
Formula :-
P(Y|X) = P(X|Y) * P(Y) / Normalization
Where P(Y) = Prior Probability
P(X|Y) = Likelihood
First calculate feature of input image
We will have then F1 to F49 of single image
Let's say F1=7
Then calculate given F1=7; how many images from dataset have F1 as 7

So, P(F1=7|Y=1)

method: calculatePrior:
    params:
        file_path_labels = training data's labels to calculate prior probability.
        value = which label's prior probability you want to find
"""

def calculatePrior(file_path_labels, value):
    count_dict = {}
    
    for i in range(10):
        count_dict[i] = 0
    
    label_file = open(file_path_labels)
    label_lines = label_file.readlines()

    for i in range(len(label_lines)):
        current = int(label_lines[i].strip())
        count_dict[current ] = count_dict[current ] + 1

    total_count = sum(count_dict.values())
    value_count = count_dict[value]
    
    return value_count, total_count

"""
Training should return me data structure which will help me build 
the likelihood probability very efficiently.

Structure would be as following:
    {label : all the possible features : possible values each feature can take : count }

Params:
    dataWithLabel: dataset(Training dataset with labels and features)
    allLabels: Number of possible labels (distinct labels)
    totalFeatures: number of features image has
    length, width : dimension of image(pixel format)
"""

def training_Bayesian(dataWithLabel, allLabels, feature_dim, length, width):
    row_ft = width/feature_dim[0]
    column_ft = length/feature_dim[0]
    
    totalFeatures = int(row_ft * column_ft)
    possibleValues = pow(feature_dim[0], 2)
    
    training_Data = {}
    for i in range(allLabels):
        featureDict = {}
        for j in range(1, totalFeatures+1):
            possibleValues_Dict = {}
            for k in range(0, possibleValues+1):
                possibleValues_Dict[k] = 0
            featureDict[j] = possibleValues_Dict
        training_Data[i] = featureDict

    for i in range(len(dataWithLabel)):
        temp_Data = dataWithLabel[i]
        temp_features = temp_Data['features']
        temp_label = int(temp_Data['label'])
    
        for j in range(len(temp_features)):
            cell_count = temp_features[j][0] + temp_features[j][1]
            count = training_Data[temp_label][j+1][cell_count]
            count = count + 1
            training_Data[temp_label][j+1][cell_count] = count
            
    return training_Data

def posteriorProbability(dataWithLabel, allLabels, prior_prob, trainingDict):
    predicted_value = []
    
    for i in range(len(dataWithLabel)):
        test = dataWithLabel[i]
        feature_test = test['features']
        
        likelihood = {}
        final_prob = {}
        
        for each_label in range(allLabels):
            probability = 1
            for each_feature in range(len(feature_test)):
                feature_sum = int(feature_test[each_feature][0] + feature_test[each_feature][1])
                val = trainingDict[each_label][each_feature+1][feature_sum] / prior_prob[each_label][0]
                if(val!=0.0):
                    probability = probability * val
                else:
                    probability = probability * 0.001
            likelihood[each_label] = probability
        
        alpha = sum(likelihood.values())
        for index in range(len(likelihood)):
            prior = (prior_prob[index][0] / prior_prob[index][1])
            final_prob[index] = (likelihood[index] / alpha) * prior
        
        predicted_value.append(max(final_prob, key = final_prob.get))
    
    return predicted_value