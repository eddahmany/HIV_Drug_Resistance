import math
import time
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt



def initialize_parameters_deep(layer_dims):

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * sqrt(1/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))


    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def sigmoid(Z):
    A = 1/ ( 1 + np.exp(-Z))

    return A, Z

def deriv_sigmoid(Z):
    sigmoid_A = sigmoid(Z)[0]
    return sigmoid_A * (1 - sigmoid_A)
def deriv_relu(Z):
    return np.where(Z > 0, 1, 0)

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache



def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):

    m = Y.shape[1]

    epsilon = 1e-8

    cost = -1 / m * (np.dot(Y, np.log(AL + epsilon).T) + np.dot((1 - Y), np.log(1 - AL + epsilon).T))

    cost = np.squeeze(cost)  # ex : [[17]] -------> 17

    return cost


def linear_backward(dZ, cache, lambd):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T) + lambd/m * W
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, lambd):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = dA * deriv_relu(activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    elif activation == "sigmoid":
        dZ = dA* deriv_sigmoid(activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    return dA_prev, dW, db



def update_parameters(params, grads, learning_rate, iterations_rate):

    parameters = params.copy()
    L = len(parameters) // 2
    if iterations_rate < 2 / 3:
        coeff = learning_rate
    else :
        coeff = learning_rate/3
    for l in range(L):

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - coeff * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - coeff * grads["db" + str(l + 1)]

    return parameters

def plot_cost_iterations(file_path):
    iterations = []
    costs = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Cost after iteration'):
                iteration, cost = line.strip().split(': ')
                iteration = int(iteration.split()[3])
                cost = float(cost)
                iterations.append(iteration)
                costs.append(cost)

    plt.plot(iterations, costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iterations')
    plt.show()


def compute_cost_with_regularization(AL, Y, parameters, lambd):

    m = Y.shape[1]
    S = 0
    for l in range(1 , AL.shape[0]) :
        S += np.sum(np.square(parameters["W"+str(l)]))

    cross_entropy_cost = compute_cost(AL, Y)

    L2_regularization_cost = lambd / (2 * m) * S


    cost = cross_entropy_cost + L2_regularization_cost

    return cost
def L_model_backward(AL, Y, caches, lambd):

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    epsilon = 1e-8
    dAL = - (np.divide(Y,  AL + epsilon) - np.divide(1 - Y, 1 - AL + epsilon  ))
    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid", lambd)
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu", lambd)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    np.random.seed(1)

    cache = {}
    A = X
    for l in range(1, (len(parameters) // 2) ):

        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        Z = np.dot(W, A) + b
        A = relu(Z)[0]

        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int)
        A *= D
        A /= keep_prob

        cache["Z" + str(l)] = Z
        cache["D" + str(l)] = D
        cache["A" + str(l)] = A
        cache["W" + str(l)] = W
        cache["b" + str(l)] = b

    W = parameters["W" + str(len(parameters) // 2)]
    b = parameters["b" + str(len(parameters) // 2)]
    Z = np.dot(W, A) + b
    A = sigmoid(Z)[0]

    cache["Z" + str(len(parameters) // 2)] = Z
    cache["A" + str(len(parameters) // 2)] = A
    cache["W" + str(len(parameters) // 2)] = W
    cache["b" + str(len(parameters) // 2)] = b

    return A, cache

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]

    L = (len(cache)+1) // 5
    gradients = {}

    dZ = cache["A" + str(L)] - Y
    dW = 1. / m * np.dot(dZ, cache["A" + str(L-1)].T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(cache["W" + str(L)].T, dZ)

    gradients["dZ" + str(L)] = dZ
    gradients["dW" + str(L)] = dW
    gradients["db" + str(L)] = db
    gradients["dA" + str(L-1)] = dA_prev

    for l in range(L-1, 0, -1):

        dA = np.dot(cache["W" + str(l+1)].T, gradients["dZ" + str(l+1)])
        dA *= cache["D" + str(l)]  # Dropout
        dA /= keep_prob
        dA = np.multiply(dA, np.int64(cache["A" + str(l)] > 0))
        dZ = dA
        if l == 1 :
            dW = 1. / m * np.dot(dZ, X.T)
        else :
            dW = 1. / m * np.dot(dZ, cache["A" + str(l-1)].T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)

        if l > 1:
            dA_prev = np.dot(cache["W" + str(l)].T, dZ)
            gradients["dA" + str(l-1)] = dA_prev

        gradients["dZ" + str(l)] = dZ
        gradients["dW" + str(l)] = dW
        gradients["db" + str(l)] = db

    return gradients


def update_parameters_with_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
        parameters["W" + str(l)] -= learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * v["db" + str(l)]


    return parameters, v


def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
        v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))


    return v


def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
        v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))
        s["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
        s["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))


    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]


        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)


        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)

        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)

        parameters["W" + str(l)] -= learning_rate * (
                    v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon))
        parameters["b" + str(l)] -= learning_rate * (
                    v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon))


    return parameters, v, s, v_corrected, s_corrected
def decay_name(decay):
    if decay == schedule_lr_decay:
        return 'schedule_lr_decay'
    elif decay == update_lr:
        return 'update_lr'
    else : return


def L_layer_model(X, Y, layers_dims, optimizer, learning_rate, num_iterations, name_file, beta = 0.9,
                  beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, print_cost=False, keep_prob=1, lambd=0, decay=None, decay_rate=1,
                  ):
    start_time = time.time()
    print("Training the model :")
    np.random.seed(1)
    costs = []  # keep track of cost
    t = 0
    learning_rate0 = learning_rate
    parameters = initialize_parameters_deep(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    path_file = '/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Results/MLP/'

    with open(path_file + name_file, 'a') as file:
        file.write('Parameters :\n')
        parameters_title = "\tLayer dimensions :"+ str(layers_dims)+ "\n\tOptimizer :" + str(optimizer) + "\n\tLearning rate : " + str(learning_rate) + "\n\tNumber of Iterations : "+str(num_iterations) + "\n\tDecay Function :" + str(decay_name(decay)) +'\n\tL2 Regulization parameter :'+ str(lambd)+ '\n\n'
        file.write(parameters_title)
        file.write('Costs :\n')

        # Loop (gradient descent)
        iterations_rate = 0
        for i in range(0, num_iterations):

            if keep_prob == 1:
                AL, caches = L_model_forward(X, parameters)
            elif keep_prob < 1:
                AL, caches = forward_propagation_with_dropout(X, parameters, keep_prob)
            cost = compute_cost(AL, Y)
            if lambd == 0:
                cost = compute_cost(AL, Y)
            else:
                cost = compute_cost_with_regularization(AL, Y, parameters, lambd)

            assert (lambd == 0 or keep_prob == 1)

            if keep_prob == 1:
                grads = L_model_backward(AL, Y, caches, lambd)
            else:
                grads = backward_propagation_with_dropout(AL, Y, caches, keep_prob)


            iterations_rate = i / num_iterations

            #old version : parameters = update_parameters(parameters, grads, learning_rate, iterations_rate)


            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                                     t, learning_rate, beta1, beta2, epsilon)

            if decay:
                learning_rate = decay(learning_rate0, i, decay_rate)
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                cost_str = "\tCost after iteration {}: {}".format(i, np.squeeze(cost))
                print(cost_str)
                if i % (num_iterations - 1) == 0 :
                    file.write(cost_str + '\n')
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)
        end_time = time.time()

        execution_time = end_time - start_time
        execution_time_minutes = execution_time / 60
        file.write("\nTraining time : {:.2f} minutes\n".format(execution_time_minutes))
        return parameters, costs


def update_lr(learning_rate0, epoch_num, decay_rate):

    learning_rate = 1 / (1 + decay_rate * (epoch_num / 1)) * learning_rate0

    return learning_rate


def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=300):

    learning_rate = 1 / (10 + decay_rate * math.floor(epoch_num / time_interval)) * learning_rate0

    return learning_rate

def predict(parameters, X):

    AL, cache = L_model_forward(X, parameters)
    predictions = (AL > 0.5)
    return np.array(predictions)

