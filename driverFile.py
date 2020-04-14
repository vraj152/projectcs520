import readingData as rd

file_path = r'data/digitdata/trainingimages'
length, width = 28, 28


loaded_data = rd.load_data(file_path, 5000, length, width)

matrices = rd.matrix_transformation(loaded_data, length, width)
