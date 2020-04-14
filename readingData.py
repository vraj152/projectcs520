import numpy as np

"""
This method reads entire file and returns as a list
Params:
    souce_file = file path
    total_images = dataset size
    length = length of pixel
    width = length of pixel
Returns:
    Entire file in list
"""

def load_data(source_file, total_images, length, width):
    
    digitFile = open(source_file)
    data_line = digitFile.readlines()
    digit_data= []

    for i in range(total_images):
        temp_data = []
        for j in range(length*i, length*(i+1)):
            temp_data.append(data_line[j])
        digit_data.append(temp_data)
        
    return digit_data


"""
This method requires entire data passed in list, and will return list of numpy array
Params:
    digit_data = list
    length = length of pixel
    width = length of pixel
Returns:
    List of Numpy array
    And array is representation of given data
"""

def matrix_transformation(digit_data, length, width):
    total_data = len(digit_data)
    final_data = []

    for i in range(total_data):
        mat = np.zeros((length, width))
        single_image = digit_data[i]
        single_image_length = len(single_image)
    
        for j in range(single_image_length):
            single_line = single_image[j]
            single_line_length = len(single_line)
        
            for k in range(single_line_length):
                if(single_line[k] == '+'):
                    mat[j][k] = 1
                if(single_line[k] == '#'):
                    mat[j][k] = 2
        final_data.append(mat)   
        
    return final_data


"""
TESTER FUNCTION - NEESDS TO BE DELETED BY END
"""
def matrix_transformation_test(digit_data, length, width):
    single_data_length = len(digit_data)

    mat = np.zeros((length, width))

    for j in range(single_data_length):
        single_line = digit_data[j]
        single_line_length = len(single_line)
    
        for k in range(single_line_length):
            if(single_line[k] == '+'):
                mat[j][k] = 1
            if(single_line[k] == '#'):
                mat[j][k] = 2
        
    return mat