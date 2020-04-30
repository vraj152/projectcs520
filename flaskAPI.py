from flask import Flask,request
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DIMENSIONS_DIGIT = (28, 28)
FEATURE_DIGIT_TUPLE = (1, 1)
models = {
    'perceptron' : 'percep_digitdata.pkl',
    'mira' : 'mira_digitdata.pkl',
    'bayesian' : ['bayes_likelihood_digitdata.pkl', 'bayes_prior_digitdata.pkl']
    } 
path = r"models/"

@app.route('/recognize', methods=['GET'])
def home():
    raw_input = eval("[" + request.args.get('imgdata') + "]")
    algoName = request.args.get('algo')
    features = manipulateList(raw_input)
    
    if(algoName == 'bayesian'):
        likelihoodData = pickle.load(open(path+models[algoName][0],"rb"))
        priorData = pickle.load(open(path+models[algoName][1],"rb"))
        
        predicted_label = posteriorProbability(features, priorData, likelihoodData)
        return str(predicted_label)
        
    else:
        model = pickle.load(open(path+models[algoName],"rb"))    
    
        dot_product_test = np.dot(model.T, features)
    
        predicted_label = np.argmax(dot_product_test)
        return str(predicted_label)

def manipulateList(raw_input):
    for i in range(len(raw_input)):
        if(raw_input[i]==(-1)):
            raw_input[i]=0
        else:
            raw_input[i]=1
    
    temp_arr = np.asarray(raw_input)
    mat_t = np.resize(temp_arr,(28, 28))
    mat = mat_t.T
    raw_input_final = []
    for i in range(len(mat)):
        for j in range(len(mat)):
            raw_input_final.append(mat[i][j])
    final_arr = np.asarray(raw_input_final)
    return final_arr

def posteriorProbability(features, prior_prob, trainingDict):
    
    likelihood = {}
    final_prob = {}
    
    for each_label in range(10):
        probability = 1
        for each_feature in range(len(features)):
            val = trainingDict[each_label][each_feature+1][features[each_feature]] / prior_prob[each_label][0]
            if(val!=0.0):
                probability = probability * val
            else:
                probability = probability * 0.001
        likelihood[each_label] = probability
    
    alpha = sum(likelihood.values())

    for index in range(len(likelihood)):
        prior = (prior_prob[index][0] / prior_prob[index][1])
        final_prob[index] = (likelihood[index] / alpha) * prior
    
    return (max(final_prob, key = final_prob.get))

if __name__ == "__main__":
    app.run(host='0.0.0.0')