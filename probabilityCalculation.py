import featureExtraction as fext
import readingData as rd

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
    
    probability = value_count / total_count
    return probability

calculatePrior(1)