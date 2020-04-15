import featureExtraction as fext
import readingData as rd
import driverFile as df

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
                        
"""
#%%
dataWithLabel = df.createDataWithLabel()
#%%

def calculatePrior(value):
    file_path_labels = r'data/digitdata/traininglabels'
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

def likelihood(ftr_indx, ftr_val, label, total_label):
    count = 0
    against = 0
    
    for i in range(len(dataWithLabel)):
        curr_pointer = dataWithLabel[i]
        if (int(curr_pointer['label']) == label):
            
            raw_info = curr_pointer['features'][ftr_indx]
            curr_feature = int(raw_info[0] + raw_info[1])
            
            if (curr_feature == ftr_val):
                count = count + 1
            else:
                against = against + 1
    if(count!=0):
        likelihood_probability = count / total_label
    else:
        #Smoothing
        likelihood_probability = 0.001

    return likelihood_probability
