import readingData as rd
import featureExtraction as fext

def createDataWithLabel():
    file_path_images = r'data/digitdata/trainingimages'
    file_path_labels = r'data/digitdata/traininglabels'
    
    length, width = 28, 28
    
    loaded_data = rd.load_data(file_path_images, 5000, length, width)
    loaded_labels = rd.load_label(file_path_labels)
    
    matrices = rd.matrix_transformation(loaded_data, length, width)
    
    dataWithLabel = {}
    
    for each_matrix in range(len(matrices)):
        temp = {}
        test = matrices[each_matrix]
        
        label = loaded_labels[each_matrix]
        feature = fext.all_at_once(test)
        
        temp = {'label' : label, 'features' : feature}
        dataWithLabel[each_matrix] = temp
    return dataWithLabel