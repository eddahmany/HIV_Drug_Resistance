import numpy as np


def compare(value):
    if value < 3.5:
        return 0
    else:
        return 1
def read_dataset(file, consensus, index_labels):
    print(">>>>>> Reading Dataset :")

    features = []
    labels =[]

    with open(file, 'r') as file:
        i = 0
        next(file) # skip the first line
        for line in file :
            #if len(features) == 200:
            #    break
            line = line.strip().split('\t')
            # Dropping NA features :
            if line[1 + index_labels] != 'NA':
                labels.append(float(line[1 + index_labels]))
                if len(consensus) == 99:
                    p_values = line[9:108]  # Extract values from P1 to P99 (9:108 to include P1 to P99)
                elif len(consensus) == 288:
                    p_values = line[6: 6 + 288]  # Extract values from P1 to P288

                encoded_sequence = np.array([ord(c) for c in consensus])
                for i, value in enumerate(p_values):
                    # Encoding :
                    if len(value) != 1:
                        encoded_sequence[i] = ord(value[0]) + ord(value[1])
                    else :
                        if value != '-':
                            encoded_sequence[i] = ord(value)
                features.append(encoded_sequence)
            #i+=1
    labels = np.where(np.array(labels) < 3.5 , 0, 1)
    labels = labels.reshape(1, labels.shape[0])
    features = np.array(features).T
    return features, labels



def compute_label_percentage(labels):
    m = labels.shape[1]
    count_1 = np.sum(labels == 1)
    count_0 = np.sum(labels == 0)

    pourcentage_1 = (count_1 / m) * 100
    pourcentage_0 = (count_0 / m) * 100

    return pourcentage_1, pourcentage_0


def split_dataset(X, Y, test_size=0.2):
    print(">>>>>> Spliting Dataset :")
    m = X.shape[-1]  # Nombre d'exemples dans le dataset
    num_test = int(m * test_size)  # Nombre d'exemples dans l'ensemble de test
    num_train = m - num_test  # Nombre d'exemples dans l'ensemble d'entraînement

    indices = np.random.permutation(m)  # Mélange aléatoire des indices
    #print(">>>>>> permutation des indices: "+str(indices))
    test_indices = indices[:num_test]  # Indices pour l'ensemble de test
    train_indices = indices[num_test:]  # Indices pour l'ensemble d'entraînement
    #print(">>>>>>indices de test : "+str(test_indices))
    #print(">>>>>>indices de train : "+str(train_indices))

    X_train = X[:,train_indices]  # Ensemble d'entraînement
    X_test = X[:,test_indices]  # Ensemble de test
    print(">>>>>>Test Set 20% (size) : "+str(X_test.shape))
    print(">>>>>>Train Set 80% (Size): "+str(X_train.shape))

    Y_train = Y[:,train_indices.reshape(-1)]  # Étiquettes d'entraînement
    Y_test = Y[:,test_indices.reshape(-1)]  # Étiquettes de test

    return X_train, X_test, Y_train, Y_test


def accuracy(predictions, labels):
    m = predictions.shape[-1]  # nombre d'exemples
    #print(">>>>>>>>>>>>>> m = "+str(m))
    correct_predictions = np.sum(predictions == labels)  # compte le nombre de prédictions correctes
    accuracy = (correct_predictions / m) *100  # calcule la précision en pourcentage
    return accuracy


#Consensus B Amino Acid Sequences
integrase_consensus = "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGKIILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED"

protease_consensus = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

if __name__ == "__main__":
    print(len(integrase_consensus))

    encoded_sequence = np.array([ord(c) for c in protease_consensus])
    #print("PI encoded_sequence :"+str(encoded_sequence))

    print("\nPI drug class :")
    print("\nFPV drug :")
    FPV_features, FPV_labels = read_dataset("PI_DataSet.txt", protease_consensus, 0)
    print( "size of FPV_features :"+str(FPV_features.shape))
    print( "size of  FPV_labels :"+str(FPV_labels.shape))
    percentage_1, percentage_0 = compute_label_percentage(FPV_labels)
    print("percentage of resistant sequences : {:.2f}%".format(percentage_1))
    print("percentage of non-resistant sequences : {:.2f}%".format(percentage_0))

    print("\nATV drug :")
    ATV_features, ATV_labels = read_dataset("PI_DataSet.txt", protease_consensus, 1)
    print( "size of ATV_features :"+str(ATV_features.shape))
    print( "size of  ATV_labels :"+str(ATV_labels.shape))
    percentage_1, percentage_0 = compute_label_percentage(ATV_labels)
    print("percentage of resistant sequences : {:.2f}%".format(percentage_1))
    print("percentage of non-resistant sequences : {:.2f}%".format(percentage_0))


    print("\nINI drug class :")
    print("\nRAL drug :")
    RAL_features, RAL_labels = read_dataset("INI_DataSet.txt", integrase_consensus, 0)
    print( "size of RAL_features :"+str(RAL_features.shape))
    print( "size of  RAL_labels :"+str(RAL_labels.shape))
    percentage_1, percentage_0 = compute_label_percentage(RAL_labels)
    print("percentage of resistant sequences : {:.2f}%".format(percentage_1))
    print("percentage of non-resistant sequences : {:.2f}%".format(percentage_0))




RT = "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVLP" \
     "QGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKEPPFLWMGYELHPDKWTVQPIVLPEKD" \
     "SWTVNDIQKLVGKLNWASQIYAGIKVKQLCKLLRGTKALTEVIPLTEEAELELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLK" \
     "TGKYARMRGAHTNDVKQLTEAVQKIATESIVIWGKTPKFKLPIQKETWEAWWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIVGAETFYVDGAANRET" \
     "KLGKAGYVTDRGRQKVVSLTDTTNQKTELQAIHLALQDSGLEVNIVTDSQ" \
     "YALGIIQAQPDKSESELVSQIIEQLIKKEKVYLAWVPAHKGIGGNEQVDKLVSAGIRKVL"
#print(len(RT))