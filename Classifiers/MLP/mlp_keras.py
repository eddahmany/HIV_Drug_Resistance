import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from Data import dataset as data
from keras.regularizers import l2

from Data.feature_importance import permutation_importance_2d

import tensorflow.keras.backend as K
from Data.feature_importance import permutation_importance


def f1_metric(y_true, y_pred):
    y_pred_classes = K.round(y_pred)
    y_true_float = K.cast(y_true, dtype=tf.float32)
    y_pred_classes_float = K.cast(y_pred_classes, dtype=tf.float32)

    tp = K.sum(y_true_float * y_pred_classes_float)
    tn = K.sum((1 - y_true_float) * (1 - y_pred_classes_float))
    fp = K.sum((1 - y_true_float) * y_pred_classes_float)
    fn = K.sum(y_true_float * (1 - y_pred_classes_float))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

def custom_loss(y_true, y_pred, aditional_arg):
    y_true = tf.cast(y_true, dtype=tf.float32)
    non_resistant_penalty = 1.0

    resistant_penalty = aditional_arg

    loss = -tf.reduce_mean(
        y_true * tf.math.log(y_pred + 1e-8) * resistant_penalty +
        (1 - y_true) * tf.math.log(1 - y_pred + 1e-8) * non_resistant_penalty
    )

    return loss


def k_fold_cross_validation(file_name, index_drug, k = 5,batch_size = 32,learning_rate = 0.001, num_epochs = 10, printing_step=1):
    layers_dims = [300, 200, 100, 1]
    L2_lambda = 0.001
    learning_rate = learning_rate
    optimizer = 'Adam'

    X, Y = data.one_hot_encoding(file_name, index_drug)
    X = np.transpose(X)
    Y = np.transpose(Y)
    print(">>>>>>> X shape : " + str(X.shape))

    number_features = int((X.shape[1])/30)
    mean_importances = np.zeros(number_features)


    num_non_resistant = np.sum(1 - Y)
    num_resistant = np.sum(Y)
    resistant_penalty = num_non_resistant / num_resistant
    print(">>>>>>> resistant_penalty : "+str(resistant_penalty))


    accuracies = []
    f1_scores =[]


    # Get the file path for writing results :
    drug_name = data.get_drug_name(file_name, index_drug)
    drug_class = file_name.split("_")[0]
    file_path = '/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Results/MLP/keras/'+ drug_class +  "/" +drug_name + '_results.txt'

    with open(file_path , 'w') as file:
        file.write('Parameters :\n')
        settings = "\tLayer dimensions : "+ str(layers_dims)+ "\n\tOptimizer : " + str(optimizer) + "\n\tLearning rate : " + str(learning_rate) + "\n\tNumber of Epochs : "+str(num_epochs) + "\n\tMiniBatch Size : "+str(batch_size) + "\n\tL2 regularization parameter :" +str(L2_lambda)+'\n\n'
        file.write(settings)

        kf = StratifiedKFold(n_splits=k, shuffle=True)
        i = 0
        for train_index, test_index in kf.split(X, Y):
            file.write("Fold : "+str(i+1))
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[ train_index], Y[test_index]

            model = Sequential()
            model.add(
                Dense(layers_dims[0], activation='relu', kernel_regularizer=l2(L2_lambda), input_shape=((X.shape[1],))))
            model.add(Dense(layers_dims[1], activation='relu', kernel_regularizer=l2(L2_lambda)))
            model.add(Dense(layers_dims[2], activation='relu', kernel_regularizer=l2(L2_lambda)))
            model.add(Dense(layers_dims[3], activation='sigmoid'))

            model.compile(optimizer=Adam(learning_rate=0.01),
                          loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, resistant_penalty),
                          metrics=['accuracy', f1_metric])

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
            train_f1_score = history.history['f1_metric']
            val_loss = history.history['val_loss']
            val_accuracy = history.history['val_accuracy']
            val_f1_score = history.history['val_f1_metric']

            for epoch in range(len(train_loss)):
                if epoch % printing_step == 0 or epoch == (num_epochs - 1):
                    file.write(
                        "\n\t\tEpoch {}: \tloss = {:.4f}, accuracy = {:.4f}, train_f1_score = {:.4f}  \tval_loss = {:.4f}, val_accuracy = {:.4f}, val_f1_score = {:.4f}".format(
                            epoch + 1, train_loss[epoch], train_accuracy[epoch], train_f1_score[epoch], val_loss[epoch],
                            val_accuracy[epoch], val_f1_score[epoch]))

            file.write("\n\tEvaluating :")

            loss, accuracy, f1_score = model.evaluate(X_test, Y_test, verbose = 1)

            file.write("\n\t\tAccuracy: {:.4f}\n".format(accuracy))


            accuracies.append(accuracy)
            f1_scores.append(f1_score)

            importances = permutation_importance_2d(model, X_test, Y_test, iterations=1)

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
            file.write("\n\t\tF1 Score: {:.4f}\n".format(f1_score))

            i+=1

        mean_accuracies = np.mean(accuracies)
        mean_f1_score = np.mean(f1_scores)

        file.write("\nMean Validation Accuracy : {:.4f}\n".format(mean_accuracies))
        file.write("\nMean Validation F1 Score : {:.4f}\n".format(mean_f1_score))

        mean_importances /= k


        importance_df = pd.DataFrame()

        importance_df['Feature'] = [f'Feature_{col}' for col in range(1, number_features + 1)]
        importance_df['Importance'] = mean_importances
        importance_df = importance_df.sort_values('Importance', ascending=False)

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        feature_importance_file_path = '/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Feature_importance/MLP/' + drug_class + "/" + drug_name + '_results.txt'

        with open(feature_importance_file_path, 'w') as file:
            file.write(importance_df.to_string(index=False))





#Calling the previous function for datasets :

for i in range(8):
    k_fold_cross_validation('PI_DataSet.txt',i , k = 5, batch_size= 32, num_epochs= 20, printing_step=2)

for i in range(6):
    k_fold_cross_validation('NRTI_DataSet.txt',i , k = 5, batch_size= 32, num_epochs= 20, printing_step=5)

for i in range(4):
    k_fold_cross_validation('INI_DataSet.txt',i , k = 5, batch_size= 32, num_epochs= 20, printing_step=5)

for i in range(4):
    k_fold_cross_validation('NNRTI_DataSet.txt',i , k = 5, batch_size= 32, num_epochs= 20, printing_step=5)
