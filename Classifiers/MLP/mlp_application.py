import get_dataset as data
import matplotlib.pyplot as plt
import mlp_functions as mlp
import numpy as np

X, Y = data.read_dataset("PI_DataSet.txt", data.protease_consensus, 0)
print( "size of X :"+str(X.shape))
print( "size of  Y :"+str(Y.shape))

X_train, X_test, Y_train, Y_test = data.split_dataset(X, Y, test_size=0.2)

learning_rate = 0.01

layers_dims = [99,33,33,33,33,33, 1]

parameters, costs = mlp.L_layer_model(X_train, Y_train, layers_dims, optimizer = 'momentum', learning_rate = 0.01, num_iterations = 15000, print_cost = True, lambd=0.7 , decay =mlp.schedule_lr_decay)

predictions_train = mlp.predict(parameters, X_train)
#print(">>>>>>>>>> predictions : "+str(predictions))

train_accuracy = data.accuracy(predictions_train, Y_train)

print(">>>>>>>>>Accuracy on train set: {:.2f}%".format(train_accuracy))

predictions_test = mlp.predict(parameters, X_test)
#print(">>>>>>>>>> predictions : "+str(predictions))

accuracy = data.accuracy(predictions_test, Y_test)

print(">>>>>>>>>Accuracy on test set: {:.2f}%".format(accuracy))


mlp.plot_costs(costs, learning_rate)




"""
#parameters, costs = mlp.two_layer_model(X, Y, layers_dims = (n_x, n_h, n_y), num_iterations = 2, print_cost=False)
#print("Cost after first iteration: " + str(costs[0]))
parameters, costs = mlp.two_layer_model(X, Y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
plot_costs(costs, learning_rate)

predictions = mlp.predict(parameters, X)
accuracy = mlp.accuracy(predictions, Y)

print(">>>>>>>>>Précision : {:.2f}%".format(accuracy))
print(">>>>>>>>>> predictions : "+str(predictions.shape))
print(">>>>>>>>>> labels : "+str(Y.shape))

layers_dims = [99,33 ,33,33,33, 1]
parameters, costs = mlp.L_layer_model(X, Y, layers_dims, learning_rate, num_iterations = 15000, print_cost = True)

#print("Cost after first iteration: " + str(costs[0]))

mlp.plot_costs(costs, learning_rate)

predictions = mlp.predict(parameters, X)
print(">>>>>>>>>> predictions : "+str(predictions))

accuracy = mlp.accuracy(predictions, Y)

print(">>>>>>>>>Précision : {:.2f}%".format(accuracy))
"""