import numpy as np

def one_hot_encoding(file, index_labels):
    # Consensus B Amino Acid Sequences

    integrase_consensus = "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGKIILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED"

    protease_consensus = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

    RT_consensus = "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKEPPFLWMGYELHPDKWT"

    drug_class = file.split("_")[0]
    if drug_class == 'PI':
        consensus = protease_consensus
    elif drug_class == 'INI':
        consensus = integrase_consensus
    else:
        consensus = RT_consensus

    start_char = ord('A')
    end_char = ord('Z')
    letter_to_int = {chr(letter_code): idx for idx, letter_code in enumerate(range(start_char, end_char + 1))}
    letter_to_int['.'] = len(letter_to_int)
    letter_to_int['*'] = len(letter_to_int)
    letter_to_int['~'] = len(letter_to_int)
    letter_to_int['#'] = len(letter_to_int)

    print("mon dictionnaire : "+str(letter_to_int))

    n = len(letter_to_int)

    print(">>>>>> Reading Dataset :")
    file_path = str('/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Data/' + file)

    features = []
    labels = []

    with open(file_path, 'r') as file:
        i = 0
        first_line = file.readline().strip().split('\t')
        feature_name = first_line[1 + index_labels]
        print("drug class : " + drug_class)
        next(file)  # skip the first line
        for line in file:
            line = line.strip().split('\t')
            if line[1 + index_labels] != 'NA':
                labels.append(float(line[1 + index_labels]))
                if len(consensus) == 99:
                    p_values = line[9:108]  # Extract values from P1 to P99 (9:108 to include P1 to P99)
                elif len(consensus) == 288:
                    p_values = line[6: 6 + 288]  # Extract values from P1 to P288
                else:
                    if drug_class == 'NNRTI':
                        p_values = line[5: 5 + 240]
                    elif drug_class == 'NRTI':
                        p_values = line[7: 7 + 240]

                # One-hot encoding
                encoded_sequence = []
                for i, value in enumerate(p_values):
                    one_hot_vector = [0] * n
                    if value == '-':
                        one_hot_vector[letter_to_int[consensus[i]]] = 1
                    else :
                        for c in value :
                            one_hot_vector[letter_to_int[c]] = 1
                    encoded_sequence.append(one_hot_vector)
                encoded_sequence = np.array(encoded_sequence).flatten()

                features.append(encoded_sequence)

    labels = np.where(np.array(labels) < 3.5, 0, 1)
    labels = labels.reshape(1, labels.shape[0])
    features = np.array(features).T
    return features, labels

def get_drug_name(file, index_labels):
    file_path = str('/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Data/' + file)
    with open(file_path, 'r') as file:
        first_line = file.readline().strip().split('\t')
        return first_line[1 + index_labels]

def encoding_name(encoding):

    if encoding ==integer_encoding:
        return 'integer_encoding'
    else : return 'one_hot_encoding'

def integer_encoding(file, index_labels):
    # Consensus B Amino Acid Sequences

    integrase_consensus = "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGKIILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED"

    protease_consensus = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

    RT_consensus = "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKEPPFLWMGYELHPDKWT"

    drug_class = file.split("_")[0]
    if drug_class == 'PI':
        consensus = protease_consensus
    elif drug_class == 'INI':
        consensus = integrase_consensus
    else:
        consensus = RT_consensus

    print(">>>>>> Reading Dataset :")
    file_path = str('/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Data/' + file)

    features = []
    labels =[]

    with open(file_path, 'r') as file:
        i = 0
        first_line = file.readline().strip().split('\t')
        feature_name = first_line[1 + index_labels]
        print("drug class : "+drug_class)
        next(file) # skip the first line
        for line in file :
            line = line.strip().split('\t')
            if line[1 + index_labels] != 'NA':
                labels.append(float(line[1 + index_labels]))
                if len(consensus) == 99:
                    p_values = line[9:108]  # Extract values from P1 to P99 (9:108 to include P1 to P99)
                elif len(consensus) == 288:
                    p_values = line[6: 6 + 288]  # Extract values from P1 to P288
                else :
                    if drug_class =='NNRTI':
                        p_values = line[5: 5 + 240]
                    elif drug_class == 'NRTI':
                        p_values = line[7: 7 + 240]

                encoded_sequence = np.array([ord(c) for c in consensus])
                for i, value in enumerate(p_values):
                    # Encoding :
                    if len(value) != 1:
                        encoded_sequence[i] = ord(value[0]) + ord(value[1])
                    else :
                        if value != '-':
                            encoded_sequence[i] = ord(value)
                features.append(encoded_sequence)

    labels = np.where(np.array(labels) < 3.5 , 0, 1)
    labels = labels.reshape(1, labels.shape[0])
    features = np.array(features).T
    return features, labels


def integer_to_one_hot(X) :
    X = X.T
    X_one_hot = []
    start_char = ord('A')
    end_char = ord('Z')
    letter_to_int = {chr(letter_code): idx for idx, letter_code in enumerate(range(start_char, end_char + 1))}
    letter_to_int['.'] = len(letter_to_int)
    letter_to_int['*'] = len(letter_to_int)
    letter_to_int['~'] = len(letter_to_int)
    letter_to_int['#'] = len(letter_to_int)


    n = len(letter_to_int)

    m = len(X)
    for i in range(m):
        one_hot_sequence = []
        for position in X[i]:
            one_hot_vector = [0] * n
            if value == '-':
                one_hot_vector[letter_to_int[consensus[i]]] = 1
            else:
                for c in value:
                    one_hot_vector[letter_to_int[c]] = 1
            encoded_sequence.append(one_hot_vector)
        encoded_sequence = np.array(encoded_sequence).flatten()

        features.append(encoded_sequence)



def compute_label_percentage(labels):
    m = labels.shape[1]
    count_1 = np.sum(labels == 1)
    count_0 = np.sum(labels == 0)

    pourcentage_1 = (count_1 / m) * 100
    pourcentage_0 = (count_0 / m) * 100

    return pourcentage_1, pourcentage_0


def split_dataset(file_name, X, Y, test_size, encoding):
    print(">>>>>> Spliting Dataset :")
    m = X.shape[-1]
    num_test = int(m * test_size)
    num_train = m - num_test

    indices = np.random.permutation(m)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    X_train = X[:,train_indices]
    X_test = X[:,test_indices]
    path_file = '/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Results/MLP/'
    print("\tTest Set : {:.2f}%".format(100 * test_size) + " (size) : " + str(X_test.shape) + '\n')
    print("\tTrain Set : {:.2f}%".format(100 * (1 - test_size)) + " (size) : " + str(X_train.shape) + '\n\n')

    with open(path_file + file_name, 'w') as file:
        file.write('Data  :\n')
        file.write('\tEncoding type : '+ encoding_name(encoding)+'\n')
        file.write("\tTest Set : {:.2f}%".format(100*test_size)+ " (size) : "+str(X_test.shape)+'\n')
        file.write("\tTrain Set : {:.2f}%".format(100*(1-test_size))+ " (size) : "+str(X_train.shape)+'\n\n')


    Y_train = Y[:,train_indices.reshape(-1)]
    Y_test = Y[:,test_indices.reshape(-1)]

    return X_train, X_test, Y_train, Y_test


def accuracy(predictions, labels, set_type, name_file = None):
    if set_type == 0:
        dataset = "Train"
    else:
        dataset = "Test"

    m = predictions.shape[-1]
    tp = np.sum((predictions == 1) & (labels == 1))  # True Positives
    tn = np.sum((predictions == 0) & (labels == 0))  # True Negatives
    fp = np.sum((predictions == 1) & (labels == 0))  # False Positives
    fn = np.sum((predictions == 0) & (labels == 1))  # False Negatives

    accuracy = (tp + tn) / m * 100
    if name_file :
        path_file = '/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Results/MLP/' + str(name_file)
        with open(path_file, 'a') as file:
            file.write("{} Set :\n".format(dataset))
            file.write("\t\tTP TN FP FN :\n".format(tp, tn, fp, fn))
            file.write("\t\t{} {} {} {}\n".format(tp, tn, fp, fn))
            file.write("\t\tAccuracy : {:.2f}%\n".format( accuracy))

    return accuracy, tp, tn, fp, fn

def f1_score(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
def statistics(filename , drug_number):
    drug_class = filename.split("_")[0]

    integrase_consensus = "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGKIILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED"

    protease_consensus = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

    RT_consensus = "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKEPPFLWMGYELHPDKWT"

    if drug_class == 'PI':
        consensus = protease_consensus
    elif drug_class == 'INI':
        consensus = integrase_consensus
    else:
        consensus = RT_consensus

    path_file = '/home/ed-dahmany/Documents/deep_learning/HIV_Drug_Resistance/Data/' + drug_class +  '_Statistics.txt'
    with open(path_file, 'w') as file:
        file.write(drug_class + ' Statistics : \n')
    for i in range(drug_number):
        features, labels = integer_encoding(filename, i)
        drug_name =  get_drug_name(filename, i)
        with open(path_file , 'a') as file:
            file.write(drug_name+' :\n')
            file.write("size of features :" + str(features.shape) + '\n')
            file.write("size of  labels :" + str(labels.shape) + '\n')
            percentage_1, percentage_0 = compute_label_percentage(labels)
            file.write("percentage of resistant sequences : {:.2f}%".format(percentage_1) + '\n')
            file.write("percentage of non-resistant sequences : {:.2f}%".format(percentage_0) + '\n')



if __name__ == "__main__":
    statistics("PI_DataSet.txt", 8)
    statistics("INI_DataSet.txt", 5)
    statistics("NRTI_DataSet.txt", 6)
    statistics("NNRTI_DataSet.txt", 4)








