import numpy as np

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

#This method requires entire data passed in list, and will return 
#list of numpy array

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
    