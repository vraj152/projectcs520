import readingData as rd
import featureExtraction as fext

"""
Created processed data:
    from the images and labels
    conatins features as well

    params:
        phase: either training or testing
            (While testing we don't need labels, so won't include them)
        file_path_images: path of input image(again either training or testing)
        loaded_labels : Corresponding labels
        total_images : Total number of image
        feature_dim : Dimensions of an input image
            (e.g. Face -> (60,70) )
        length, width : of an image
    returns:
        dictionary of above said information
        
"""
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