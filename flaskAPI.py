from flask import Flask,request
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DIMENSIONS_DIGIT = (28, 28)
FEATURE_DIGIT_TUPLE = (1, 1)

path = r"C:/Users/Dell/projectcs520/models/percep_digitdata.pkl"
model = pickle.load(open(path,"rb"))

@app.route('/recognize', methods=['GET'])
def home():
    raw_input = eval("[" + request.args.get('imgdata') + "]")
    features = manipulateList(raw_input)
    dot_product_test = np.dot(model.T, features)
    predicted_label = np.argmax(dot_product_test)
    print(predicted_label)
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

if __name__ == "__main__":
    app.run(host='0.0.0.0')