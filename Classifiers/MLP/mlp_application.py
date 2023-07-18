

from Data import dataset as data

import matplotlib.pyplot as plt
import mlp_functions as mlp
import numpy as np
import time


from Data.dataset import protease_consensus

from Data.dataset import integrase_consensus

from Data.dataset import RT_consensus

learning_rate = 0.01
optimizer = 'adam'
decay = mlp.schedule_lr_decay
layers_dims = [33,33,33,33, 1]
num_iterations = 20000
lambd=0.8

def apply_mlp_on_hiv_dataset(drugs_number, consensus, file):
    for i in range (1, drugs_number):
        start_time = time.time()
        X, Y , feature_name = data.read_dataset(file, consensus, i)
        print( "size of X :"+str(X.shape))
        print( "size of  Y :"+str(Y.shape))
        print("feature name :"+str(feature_name))
        drug_class = file.split("_")[0]
        file_path = drug_class + "/" + feature_name + '_results.txt'

        X_train, X_test, Y_train, Y_test = data.split_dataset(X, Y, 0.1, file_path)

        layers_dims.insert(0,X.shape[0])

        parameters, costs = mlp.L_layer_model(X_train, Y_train, layers_dims, optimizer = optimizer, learning_rate = learning_rate, num_iterations =num_iterations,name_file = file_path,  print_cost = True,keep_prob = 1,  lambd=0.8 , decay =decay )

        predictions_train = mlp.predict(parameters, X_train)

        train_accuracy = data.accuracy(predictions_train, Y_train, file_path, 0)[0]

        print(">>>>>>>>>Accuracy on train set: {:.2f}%".format(train_accuracy))

        predictions_test = mlp.predict(parameters, X_test)

        test_accuracy = data.accuracy(predictions_test, Y_test,file_path, 1)[0]

        print(">>>>>>>>>Accuracy on test set: {:.2f}%".format(test_accuracy))

        layers_dims.pop(0)



apply_mlp_on_hiv_dataset(8, protease_consensus, 'PI_DataSet.txt')
apply_mlp_on_hiv_dataset(4, protease_consensus, 'NNRTI_DataSet.txt')
apply_mlp_on_hiv_dataset(5, protease_consensus, 'INI_DataSet.txt')
apply_mlp_on_hiv_dataset(6, protease_consensus, 'NRTI_DataSet.txt')



