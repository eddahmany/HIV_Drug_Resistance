import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, SimpleRNN
from keras.src.legacy_tf_layers.core import Flatten
from keras.src.optimizers import RMSprop
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from Data import dataset as data
from keras.regularizers import l2

from Data.feature_importance import permutation_importance


def brnn(file_name, index_drug, k = 5,learning_rate = 0.01, batch_size = 32, num_epochs = 10, printing_step=1):

    # Get the file path for writing results :
    drug_name = data.get_drug_name(file_name, index_drug)
    drug_class = file_name.split("_")[0]
    file_path = '/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Results/BRNN/'+ drug_class +  "/" +drug_name + '_results.txt'

    L2_lambda = 0
    dropout_rate = 0.1

    optimizer = 'Adam'

    X, Y = data.one_hot_encoding(file_name, index_drug)
    X = np.transpose(X)
    print(">>>>>>> shape of X :"+str(X.shape))
    Y = np.transpose(Y)
    X = X.reshape(-1, int(X.shape[1] / 30), 30)

    accuracies = []
    mean_importances = np.zeros((X.shape[1]))



    with open(file_path , 'w') as file:
        file.write('Parameters :\n')
        settings = "\tOptimizer : " + str(optimizer) + "\n\tLearning rate : " + str(learning_rate) + "\n\tNumber of Epochs : "+str(num_epochs) + "\n\tMiniBatch Size : "+str(batch_size) + "\n\tRegularization parameters :"+'\n\t\tL2 lambda : '+str(L2_lambda)+'\n\t\tDropout rate :'+str(dropout_rate)+'\n\n'
        file.write(settings)
        if k > 1 :
            kf = KFold(n_splits=k, shuffle=True)
            indexes = kf.split(X)
        else :
            m = X.shape[0]
            all_indices = np.arange(m)
            np.random.shuffle(all_indices)
            test_size = int(m * 0.1)
            test_index = all_indices[:test_size]
            train_index = all_indices[test_size:]

            indexes = [(train_index, test_index)]
        i = 0
        for train_index, test_index in indexes:
            file.write("Fold : "+str(i+1))
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[ train_index], Y[test_index]
            print("X train shape : "+str(X_train.shape))
            print("X test shape : "+str(X_test.shape))

            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)

            model = Sequential()

            model.add(Bidirectional(
                LSTM(2, activation='relu', kernel_regularizer=l2(L2_lambda), input_shape=(X.shape[1], X.shape[2]))))
            model.add(Dropout(rate=dropout_rate))
            model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(L2_lambda)))
            model.compile(optimizer=Adam(learning_rate=learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            history = tf.keras.callbacks.History()


            file.write("\n\tTraining :")
            start_time = time.time()


            model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=1,
                      callbacks=[history])
            end_time = time.time()

            execution_time = end_time - start_time

            minutes = execution_time // 60
            seconds = execution_time % 60

            execution_time_str = f"{int(minutes)} min {seconds:.2f} s"

            file.write("\tTraining time : " + execution_time_str + "\n\n")
            train_loss = history.history['loss']
            train_accuracy = history.history['accuracy']
            val_loss = history.history['val_loss']
            val_accuracy = history.history['val_accuracy']

            for epoch in range(len(train_loss)):
                if epoch % printing_step == 0 or epoch==(num_epochs - 1):
                    file.write(
                "\n\t\tEpoch {}: \tloss = {:.4f}, accuracy = {:.4f}, \tval_loss = {:.4f}, val_accuracy = {:.4f}".format(
                    epoch + 1, train_loss[epoch], train_accuracy[epoch], val_loss[epoch], val_accuracy[epoch]))

            file.write("\n\tEvaluating :")

            loss, accuracy = model.evaluate(X_test, Y_test, verbose = 1)

            file.write("\n\t\tLoss: {:.4f}\n".format(loss))

            accuracies.append(accuracy)

            importances = permutation_importance(model, X_test, Y_test, iterations=1)

            mean_importances += importances

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

        importance_df = pd.DataFrame()

        importance_df['Feature'] = [f'Feature_{col}' for col in range(1, (X.shape[1] + 1))]
        importance_df['Importance'] = mean_importances
        importance_df = importance_df.sort_values('Importance', ascending=False)

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        feature_importance_file_path = '/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Feature_importance/BRNN/' + drug_class + "/" + drug_name + '_results.txt'

        with open(feature_importance_file_path, 'w') as file:
            file.write(importance_df.to_string(index=False))



for i in range(8):
    brnn('PI_DataSet.txt', i, k = 5,learning_rate = 0.01, num_epochs= 30, printing_step= 2)


for i in range(4):
    brnn('INI_DataSet.txt', i, k = 5,learning_rate = 0.125, num_epochs= 30, printing_step= 2)


for i in range(6):
    brnn('NRTI_DataSet.txt', i, k = 5,learning_rate = 0.1, num_epochs= 30, printing_step= 2)


for i in range(4):
    brnn('NNRTI_DataSet.txt', i, k = 5,learning_rate = 0.1, num_epochs= 30, printing_step= 2)
