import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from Data import dataset as data
from keras.regularizers import l2

def k_fold_cross_validation(file_name, index_drug, k = 5,batch_size = 32, num_epochs = 10, printing_step=1):
    layers_dims = [300, 200, 100, 1]
    L2_lambda = 0
    learning_rate = 0.01
    optimizer = 'Adam'

    X, Y = data.one_hot_encoding(file_name, index_drug)
    X = np.transpose(X)
    Y = np.transpose(Y)
    accuracies = []

    # Get the file path for writing results :
    drug_name = data.get_drug_name(file_name, index_drug)
    drug_class = file_name.split("_")[0]
    file_path = '/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Results/MLP/cross_validation/'+ drug_class +  "/" +drug_name + '_results.txt'

    with open(file_path , 'w') as file:
        file.write('Parameters :\n')
        settings = "\tLayer dimensions : "+ str(layers_dims)+ "\n\tOptimizer : " + str(optimizer) + "\n\tLearning rate : " + str(learning_rate) + "\n\tNumber of Epochs : "+str(num_epochs) + "\n\tMiniBatch Size : "+str(batch_size) + "\n\tL2 regularization parameter :" +str(L2_lambda)+'\n\n'
        file.write(settings)

        kf = KFold(n_splits=k, shuffle=True)
        i = 0
        for train_index, test_index in kf.split(X):
            file.write("Fold : "+str(i+1))
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[ train_index], Y[test_index]
            model = Sequential()
            model.add(Dense(layers_dims[0], activation='relu', kernel_regularizer=l2(L2_lambda), input_shape=(X_train.shape[1],)))
            model.add(Dense(layers_dims[1], activation='relu', kernel_regularizer=l2(L2_lambda)))
            model.add(Dense(layers_dims[2], activation='relu', kernel_regularizer=l2(L2_lambda)))
            model.add(Dense(layers_dims[3], activation='sigmoid'))

            model.compile(optimizer=Adam(learning_rate=0.01),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            history = tf.keras.callbacks.History()

            file.write("\n\tTraining :")
            start_time = time.time()
            model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=0,
                      callbacks=[history])

            train_loss = history.history['loss']
            train_accuracy = history.history['accuracy']
            val_loss = history.history['val_loss']
            val_accuracy = history.history['val_accuracy']

            for epoch in range(len(train_loss)):
                if epoch % printing_step == 0 or epoch == (num_epochs - 1):
                    file.write(
                "\n\t\tEpoch {}: \tloss = {:.4f}, accuracy = {:.4f}, \tval_loss = {:.4f}, val_accuracy = {:.4f}".format(
                    epoch + 1, train_loss[epoch], train_accuracy[epoch], val_loss[epoch], val_accuracy[epoch]))

            file.write("\n\tEvaluating :")

            loss, accuracy = model.evaluate(X_test, Y_test)

            file.write("\n\t\tAccuracy: {:.4f}\n".format(accuracy))


            accuracies.append(accuracy)

            predictions = model.predict(X_test)

            binary_predictions = np.round(predictions)


            tp = np.sum(np.logical_and(binary_predictions == 1, Y_test == 1))
            tn = np.sum(np.logical_and(binary_predictions == 0, Y_test == 0))
            fp = np.sum(np.logical_and(binary_predictions == 1, Y_test == 0))
            fn = np.sum(np.logical_and(binary_predictions == 0, Y_test == 1))

            file.write("\n\t\tTP TN FP FN :")
            file.write(str(tp)+' '+str(tn)+' '+str(fp)+' '+str(fn))
            file.write("\n\t\tAccuracy: {:.4f}\n".format(accuracy))
            i+=1

        mean_accuracies = np.mean(accuracies)
        file.write("\nMean Validation Accuracy : {:.4f}\n".format(mean_accuracies))





#Calling the previous function for PI_dataset :
for i in range(8):
    k_fold_cross_validation('PI_DataSet.txt',i , k = 5, batch_size= 32, num_epochs= 50, printing_step=5)