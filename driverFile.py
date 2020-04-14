import readingData as rd
import featureExtraction as fext

file_path = r'data/digitdata/trainingimages'
length, width = 28, 28

loaded_data = rd.load_data(file_path, 5000, length, width)

matrices = rd.matrix_transformation(loaded_data, length, width)

feature = {} 

for each_matrix in range(len(matrices)):
    test = matrices[each_matrix]
    feature[each_matrix] = fext.all_at_once(test)