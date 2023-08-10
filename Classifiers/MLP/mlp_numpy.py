

from Data import dataset as data, dataset

import matplotlib.pyplot as plt
import mlp_functions_numpy as mlp
import numpy as np
import time

from Data.dataset import encoding_name

learning_rate = 0.01
optimizer = 'adam'
decay = mlp.schedule_lr_decay
test_size = 0.1

def apply_mlp_on_hiv_dataset(file, drug_index, learning_rate, optimizer, num_iterations, encoding, printing_step = 10):

    # Get the file path for writing results :
    drug_name = data.get_drug_name(file, drug_index)
    drug_class = file.split("_")[0]
    file_path = drug_class + "/" + encoding_name(encoding) + "/" +   drug_name + '_results.txt'

    # Preparing Data :
    X, Y  = encoding(file, drug_index)
    X_train, X_test, Y_train, Y_test = data.split_dataset(file_path, X, Y, test_size , encoding)

    # Choosing layers dimensions adapted to the input dimension.
    layers_dims = [X.shape[0]]
    if encoding == dataset.one_hot_encoding:
        if drug_class == 'PI' :
            layers_dims.append(int(layers_dims[0] / 10))
            layers_dims.append(int(layers_dims[0] / 20))
            layers_dims.append(int(layers_dims[0] / 30))
        else:
            layers_dims.append(int(layers_dims[0] / 50))
            layers_dims.append(int(layers_dims[0] / 60))
            layers_dims.append(int(layers_dims[0] / 70))
    else:  # integer_encoding
        layers_dims.append(int(layers_dims[0] / 2))
        layers_dims.append(int(layers_dims[0] / 4))
        layers_dims.append(int(layers_dims[0] / 8))

    layers_dims.append(1)

    # Training The model :
    parameters = mlp.train_model(X_train, Y_train, layers_dims, optimizer = optimizer, learning_rate = learning_rate,encoding = encoding, num_iterations =num_iterations,name_file = file_path,  print_cost = True, printing_step = printing_step,keep_prob = 1,  lambd=1 , decay =decay )

    # Evaluating the model :

    mlp.evaluate(file_path, X_test, Y_test, parameters)



#Calling the function apply_mlp_on_hiv_dataset for PI drugs
for i in range(8):
    apply_mlp_on_hiv_dataset('PI_DataSet.txt',i, learning_rate,  optimizer, 10000, data.integer_encoding, printing_step= 500)
    apply_mlp_on_hiv_dataset('PI_DataSet.txt',i, learning_rate,  optimizer, 50, data.one_hot_encoding, printing_step= 10)


#Calling the function apply_mlp_on_hiv_dataset for INI drugs
for i in range(4):
    apply_mlp_on_hiv_dataset('INI_DataSet.txt',i, learning_rate,  optimizer, 10000, data.integer_encoding, printing_step= 500)
    apply_mlp_on_hiv_dataset('INI_DataSet.txt',i, learning_rate,  optimizer, 100, data.one_hot_encoding, printing_step= 10)


#Calling the function apply_mlp_on_hiv_dataset for NRTI drugs
for i in range(6):
    apply_mlp_on_hiv_dataset('NRTI_DataSet.txt',i, learning_rate,  optimizer, 10000, data.integer_encoding, printing_step= 500)
    apply_mlp_on_hiv_dataset('NRTI_DataSet.txt',i, learning_rate,  optimizer, 100, data.one_hot_encoding, printing_step= 10)
    

#Calling the function apply_mlp_on_hiv_dataset for NNRTI drugs
for i in range(4):
    apply_mlp_on_hiv_dataset('NNRTI_DataSet.txt',i, learning_rate,  optimizer, 10000, data.integer_encoding, printing_step= 500)
    apply_mlp_on_hiv_dataset('NNRTI_DataSet.txt',i, learning_rate,  optimizer, 100, data.one_hot_encoding, printing_step= 10)


