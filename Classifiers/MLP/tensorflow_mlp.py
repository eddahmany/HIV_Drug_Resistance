import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time
from Data import dataset as data

def initialize_parameters(input_dim):
    initializer = tf.keras.initializers.GlorotNormal(seed=1)

    W1 = tf.Variable(initializer(shape=(int(input_dim/20), input_dim)))
    b1 = tf.Variable(initializer(shape=(int(input_dim/20), 1)))
    W2 = tf.Variable(initializer(shape=(int(input_dim/40), int(input_dim/20))))
    b2 = tf.Variable(initializer(shape=(int(input_dim/40), 1)))
    W3 = tf.Variable(initializer(shape=(int(input_dim/80), int(input_dim/40))))
    b3 = tf.Variable(initializer(shape=(int(input_dim/80), 1)))
    W4 = tf.Variable(initializer(shape=(1, int(input_dim/80))))
    b4 = tf.Variable(initializer(shape=(1, 1)))


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4":b4}

    return parameters


def sigmoid(z):

    # tf.keras.activations.sigmoid requires float16, float32, float64, complex64, or complex128.
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)

    return a


def forward_propagation(X, parameters):


    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
                                        # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)   # Z1 = np.dot(W1, X) + b1
    A1 = tf.keras.activations.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, A1) + b2
    A2 = tf.keras.activations.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3, A2) + b3
    A3 = tf.keras.activations.relu(Z3)  # A3 = relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)  # Z4 = np.dot(W4, A3) + b4
    A4 = sigmoid(Z4)  # A4 = sigmoid(Z4)
    return A4


def compute_total_loss(logits, labels):

    # remember to set `from_logits=True`
    #print("logits shape : "+str(logits.shape))
    #print("labels shape : "+str(labels.shape))

    total_loss = tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits=False))

    return total_loss


def compute_total_loss_with_regularization(logits, labels, parameters, L2_lambda):


    # Compute the cross-entropy loss
    cross_entropy_loss =  tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits=False))


    # Compute the L2 regularization term
    l2_regularization = 0
    for param in parameters.values():
        l2_regularization += tf.reduce_sum(tf.square(param))

    # Compute the total loss with L2 regularization
    total_loss = cross_entropy_loss + (L2_lambda / (2)) * l2_regularization

    return total_loss


def model(X_train, Y_train, X_test, Y_test, file_path, L2_lambda = 0, learning_rate=0.01,
          num_epochs=1500, minibatch_size=32, print_cost=True, printing_step=10):

    costs = []  # To keep track of the cost
    train_acc = []
    test_acc = []

    # Initialize your parameters
    input_spec = X_train.element_spec

    input_size = input_spec.shape[0]
    parameters = initialize_parameters(input_size)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    layers_dims = [input_size, W1.shape[0], W2.shape[0], W3.shape[0], W4.shape[0] ]

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    test_accuracy = tf.keras.metrics.BinaryAccuracy()
    train_accuracy = tf.keras.metrics.BinaryAccuracy()

    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip((X_test, Y_test))

    m = dataset.cardinality().numpy()

    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)
    start_time = time.time()
    with open(file_path , 'w') as file:
        file.write('Settings :\n')
        settings = "\tLayer dimensions : "+ str(layers_dims)+ "\n\tOptimizer : " + str(optimizer) + "\n\tLearning rate : " + str(learning_rate) + "\n\tNumber of Epochs : "+str(num_epochs) + "\n\tMiniBatch Size : "+str(minibatch_size) + "\n\tL2 regularization parameter :" +str(L2_lambda)+'\n\n'
        file.write(settings)

        file.write('Training the model:\n')

        for epoch in range(num_epochs):

            epoch_total_loss = 0.

            # We need to reset object to start measuring from 0 the accuracy each epoch
            train_accuracy.reset_states()

            for (minibatch_X, minibatch_Y) in minibatches:
                with tf.GradientTape() as tape:
                    A4 = forward_propagation(tf.transpose(minibatch_X), parameters)
                    if L2_lambda == 0 :
                        minibatch_total_loss = compute_total_loss(A4, tf.transpose(minibatch_Y))
                    else :
                        minibatch_total_loss = compute_total_loss_with_regularization(A4, tf.transpose(minibatch_Y), parameters, L2_lambda)

                # We accumulate the accuracy of all the batches
                train_accuracy.update_state(minibatch_Y, tf.transpose(A4))

                trainable_variables = [W1, b1, W2, b2, W3, b3, W4 , b4]
                grads = tape.gradient(minibatch_total_loss, trainable_variables)
                optimizer.apply_gradients(zip(grads, trainable_variables))
                epoch_total_loss += minibatch_total_loss

            # We divide the epoch total loss over the number of samples
            epoch_total_loss /= m

            if print_cost == True and (epoch % printing_step == 0 or epoch == num_epochs - 1):
                print("Cost after epoch %i: %f" % (epoch, epoch_total_loss))
                print("Train accuracy:", train_accuracy.result())
                file.write("\n\tCost after epoch %i: %f" % (epoch, epoch_total_loss))
                file.write("\n\tTrain accuracy:"+ str(train_accuracy.result()))

                # We evaluate the test set every 10 epochs to avoid computational overhead
                for (minibatch_X, minibatch_Y) in test_minibatches:
                    A4 = forward_propagation(tf.transpose(minibatch_X), parameters)
                    test_accuracy.update_state(minibatch_Y, tf.transpose(A4))
                print("Test_accuracy:", test_accuracy.result())
                file.write("\n\tTest_accuracy:"+str(test_accuracy.result()))

                costs.append(epoch_total_loss)
                train_acc.append(train_accuracy.result())
                test_acc.append(test_accuracy.result())
                test_accuracy.reset_states()

        end_time = time.time()

        execution_time = end_time - start_time

        minutes = execution_time // 60
        seconds = execution_time % 60

        # Format the string in "X min Y s" format
        execution_time_str = f"{int(minutes)} min {seconds:.2f} s"

        file.write("\n\nTraining time : " + execution_time_str + "\n\n")

        file.write("\n\nLearned Parameters : \n\t"+ str(parameters))


        return parameters, costs, train_acc, test_acc

def split_dataset(X, Y, test_size):
    m = X.shape[-1]
    num_test = int(m * test_size)
    num_train = m - num_test

    indices = np.random.permutation(m)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    X_train = X[:,train_indices]
    X_test = X[:,test_indices]

    Y_train = Y[:,train_indices.reshape(-1)]
    Y_test = Y[:,test_indices.reshape(-1)]

    return X_train, X_test, Y_train, Y_test



def prepare_data(file_name, index_drug): #one hot encoding + converting arrays into tf.tensors
    X, Y  = data.one_hot_encoding(file_name, index_drug)

    X_train, X_test, Y_train, Y_test =split_dataset(X, Y, 0.1 )

    X_train = tf.convert_to_tensor(X_train.T, dtype=tf.float32)
    Y_train = tf.convert_to_tensor(Y_train.T, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test.T, dtype=tf.float32)
    Y_test = tf.convert_to_tensor(Y_test.T, dtype=tf.float32)

    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    X_test = tf.data.Dataset.from_tensor_slices(X_test)
    Y_train = tf.data.Dataset.from_tensor_slices(Y_train)
    Y_test = tf.data.Dataset.from_tensor_slices(Y_test)

    return X_train, X_test, Y_train, Y_test



def apply_tensorflow_mlp_on_hiv_dataset(file_name, drug_index, num_epochs = 50, learning_rate = 0.01, minibatch_size = 512):

    # Get the file path for writing results :
    drug_name = data.get_drug_name(file_name, drug_index)
    drug_class = file_name.split("_")[0]
    file_path = '/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Results/MLP/tensorflow/'+ drug_class +  "/" +drug_name + '_results.txt'

    #preparing data :
    X_train, X_test, Y_train, Y_test = prepare_data(file_name, drug_index)

    # training the model and saving results on file :
    parameters, costs, train_acc, test_acc = model(X_train, Y_train, X_test, Y_test, file_path,minibatch_size=minibatch_size, num_epochs=num_epochs, printing_step=5, L2_lambda=1, learning_rate= learning_rate)



for i in range(8):
    apply_tensorflow_mlp_on_hiv_dataset('PI_DataSet.txt', i)

for i in range(6):
    apply_tensorflow_mlp_on_hiv_dataset('NRTI_DataSet.txt', i)


for i in range(4):
    apply_tensorflow_mlp_on_hiv_dataset('NNRTI_DataSet.txt', i)

