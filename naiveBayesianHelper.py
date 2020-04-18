import probabilityCalculation as pc
import driverFile as df
import readingData as rd
import pickle

def do_training(directory, mname,pname, path):
    true_labels = rd.load_label(directory['paths'][1])
    allLabels = len(list(set(true_labels)))
    total_images = len(true_labels)

    print("===========================")
    print("Reading data")

    dataWithLabel = df.createDataWithLabel('training',directory['paths'][0], true_labels, total_images, directory['featuretuple'], directory['dimensions'][1], directory['dimensions'][0])

    print("===========================")
    print("Initializing training")

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
    print("Training done with accuracy ",accuracy)
    
    with open(path+mname,'wb') as file:
        pickle.dump(trainingDict, file)
    with open(path+pname,'wb') as file:
        pickle.dump(prior_prob, file)

def do_testing(mname,pname, directory):
    learnedLikelihood = pickle.load(open(mname,"rb"))
    priorProb = pickle.load(open(pname,"rb"))
    
    true_labels = rd.load_label(directory['paths'][3])
    allLabels = len(list(set(true_labels)))
    total_images = len(true_labels)

    print("===========================")
    print("Reading data")

    processedData = df.createDataWithLabel('testing',directory['paths'][2], true_labels, total_images, directory['featuretuple'], directory['dimensions'][1], directory['dimensions'][0])
    
    print("===========================")
    print("Initializing testing")
    
    predicted_value = pc.posteriorProbability(processedData, allLabels, priorProb, learnedLikelihood)

    diff = 0
    for i in range(len(true_labels)):
        if(int(true_labels[i]) != predicted_value[i]):
            diff = diff + 1
    
    accuracy = 100-((diff*100)/total_images)
    print("Total error in testing data: ", diff)
    print("Accuracy: ",accuracy)