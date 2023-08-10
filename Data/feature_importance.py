import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.layers import Conv1D, MaxPooling1D
from keras.src.legacy_tf_layers.core import Flatten
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from Data import dataset as data
from keras.regularizers import l2

def permutation_importance(model, X, Y, iterations = 1):
    baseline = model.evaluate(X, Y, verbose = 0)[1]
    num_features = X.shape[1]
    importances = np.zeros((num_features))
    for j in range(iterations):
        for col_idx in range(num_features):
            X_permuted = X.copy()
            print(">>>> feature importance : shape of X_permuted :" + str(X_permuted.shape))

            X_permuted[:, col_idx, :] = np.apply_along_axis(np.random.permutation, axis=0,
                                                            arr=X_permuted[:, col_idx, :])
            permuted_accuracy = model.evaluate(X_permuted, Y)[1]
            importances[col_idx] += abs(baseline - permuted_accuracy)
    importances /= iterations
    return importances

def permutation_importance_2d(model, X, Y, iterations = 1):
    baseline = model.evaluate(X, Y, verbose = 0)[1]
    num_features = int(X.shape[1] / 30)
    importances = np.zeros((num_features))
    step = 30
    for i in range(iterations):
        for col_idx in range(num_features):
            X_permuted = X.copy()
            start_idx = step * col_idx
            end_idx = start_idx + step
            X_permuted[:, start_idx: end_idx] = np.apply_along_axis(np.random.permutation, axis=0,
                                                                  arr=X_permuted[:, start_idx: end_idx])

            permuted_accuracy = model.evaluate(X_permuted, Y)[1]
            importances[col_idx] += abs(baseline - permuted_accuracy)
        importances /= iterations
        return importances

if __name__ == "__main__":

    file_name = 'PI_DataSet.txt'
    index_drug = 0
    # Get the file path for writing results :
    drug_name = data.get_drug_name(file_name, index_drug)
    drug_class = file_name.split("_")[0]
    file_path = '/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Results/CNN/'+ drug_class +  "/" +drug_name + '_results.txt'

    L2_lambda = 0.001
    learning_rate = 0.01
    optimizer = 'Adam'

    X, Y = data.one_hot_encoding(file_name, index_drug)
    X = np.transpose(X)
    print(">>>>>>> shape of X :"+str(X.shape))
    Y = np.transpose(Y)
    print(">>>>>>> X shape : "+str(X.shape))

    m = X.shape[0]
    all_indices = np.arange(m)
    np.random.shuffle(all_indices)
    test_size = int(m * 0.1)
    test_index = all_indices[:test_size]
    train_index = all_indices[test_size:]
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[ train_index], Y[test_index]
    X_train = X_train.reshape(-1, int(X_train.shape[1] / 30), 30)
    X_test = X_test.reshape(-1, int(X_test.shape[1] / 30), 30)

    print(">>>>>>> X shape : "+str(X.shape))

    accuracies = []
    mean_importances = []

    my_model = Sequential()

    my_model.add(Conv1D(64, kernel_size=9, activation='relu', input_shape=((X_train.shape[1], X_train.shape[2])),
                     kernel_regularizer=l2(L2_lambda)))
    my_model.add(MaxPooling1D(pool_size=5))
    my_model.add(Conv1D(64, kernel_size=9, activation='relu', kernel_regularizer=l2(L2_lambda)))
    my_model.add(MaxPooling1D(pool_size=5))

    my_model.add(tf.keras.layers.Flatten())
    my_model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(L2_lambda)))

    my_model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    my_model.fit(X_train, Y_train, epochs = 5, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

    importances = permutation_importance(my_model, X_test, Y_test, iterations=1)
    print("importance shape : " + str(importances.shape))
    print("importance : " + str(importances))



    # Créer un DataFrame pour afficher les résultats
    importance_df = pd.DataFrame()

    # Ajouter les noms de colonnes dans le DataFrame
    importance_df['Feature'] = [f'Feature_{col}' for col in range(1, (X_train.shape[1] + 1))]

    # Ajouter les importances calculées dans le DataFrame
    importance_df['Importance'] = importances

    # Triez les scores d'importance par ordre décroissant
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Afficher les résultats
    print(importance_df)
