import readingData as rd
import featureExtraction as fext

def createDataWithLabel(phase, file_path_images, loaded_labels, total_images,feature_dim, length, width):
    
    loaded_data = rd.load_data(file_path_images, total_images, length, width)
    matrices = rd.matrix_transformation(loaded_data, length, width)
    
    processedData = {}
    
    for each_matrix in range(len(matrices)):
        temp = {}
        test = matrices[each_matrix]
        
        label = loaded_labels[each_matrix]
        feature = fext.all_at_once(test, length, width, feature_dim)
        
        if(phase == 'training'):
            temp = {'label' : label, 'features' : feature}
        else:
            temp = {'features' : feature}
        processedData[each_matrix] = temp
        
    return processedData