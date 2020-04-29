import readingData as rd
import featureExtraction as fext
from sklearn.model_selection import train_test_split

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
def createDataWithLabel(phase, file_path_images, loaded_labels, total_images,feature_dim, length, width, splitParam):
    
    loaded_data = rd.load_data(file_path_images, total_images, length, width)
    y_train = loaded_labels
    if(phase == 'training' and splitParam!=0):
        X_train, X_test, y_train, y_test = train_test_split(loaded_data , loaded_labels, test_size=splitParam)
        matrices = rd.matrix_transformation(X_train, length, width)
    else:
        matrices = rd.matrix_transformation(loaded_data, length, width)
    
    processedData = {}
    
    for each_matrix in range(len(matrices)):
        temp = {}
        test = matrices[each_matrix]
        
        feature = fext.all_at_once(test, length, width, feature_dim)
        
        if(phase == 'training'):
            label = y_train[each_matrix]
            temp = {'label' : label, 'features' : feature}
        else:
            temp = {'features' : feature}
        processedData[each_matrix] = temp
    
    if(phase == 'training'):
        return processedData, y_train
    else:
        return processedData