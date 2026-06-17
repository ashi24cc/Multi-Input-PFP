import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from Model import nGram, dictionary, DC_CNN_Model
from sklearn.metrics import roc_auc_score
from Evaluate import recall, precision
import keras

def segment(dataset, label, seg_size, overlap, m_len):
    print("Non-overlapping Region: %s" %overlap)
    print("Segment Size: %s" %seg_size)

    seq_data, label_data = [], []
    for j, row in enumerate(dataset):
        if(len(row) < m_len + 1):
            pos = math.ceil(len(row)/overlap)
            if(pos < math.ceil(seg_size/overlap)):
                pos = math.ceil(seg_size/overlap)
            for itr in range(pos - math.ceil(seg_size/overlap) + 1):
                init = itr * overlap
                if(len(row[init : init + seg_size]) > 40):
                    seq_data.append(row[init : init + seg_size])
                    label_data.append(label[j])
    return seq_data, label_data

dataframe = pd.read_csv('mf/trainData.csv', header=None)
dataset = dataframe.values
print('Original Dataset Size : %s' %len(dataset))
X = dataset[:,0]
Y = dataset[:,1:len(dataset[0])]
nb_of_cls = len(Y[0])
del dataframe, dataset
print(X.shape, Y.shape)
print(nb_of_cls)

# Preparing For Training
segmentSize = 200
nonOL = segmentSize - 100
SEG = str(segmentSize)

# CREATING DICTIONARY
chunkSize = 4
dict_Prop = dictionary(chunkSize)
max_seq_len = segmentSize - chunkSize + 1

# Create & Compile the model
model = DC_CNN_Model(len(dict_Prop), max_seq_len, nb_of_cls)
print(model.summary())

div = [200, 500, 1000, 2000]
for max_len in div:
    X1, Y1 = segment(X, Y, segmentSize, nonOL, max_len)

    #Split the dataset
    x_tr, x_val, y_tr, y_val = train_test_split(X1, Y1, test_size = 0.1, random_state = 42)

    y_train = np.array(y_tr, dtype=None)
    y_validate = np.array(y_val, dtype=None)
    print(len(x_tr), len(x_val))
    print(y_train.shape, y_validate.shape)

    del y_tr, y_val

    #CREATING N-GRAM
    x_train = nGram(x_tr, chunkSize, dict_Prop)
    x_validate = nGram(x_val, chunkSize, dict_Prop)
    #del x_tr, x_val

    # truncate and pad input sequences
    x_train = sequence.pad_sequences(x_train, maxlen=max_seq_len)
    x_validate = sequence.pad_sequences(x_validate, maxlen=max_seq_len)

    # Train
    early_stopping_monitor1 = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
    history = model.fit(x_train, y_train.astype(None),
                        validation_data = (x_validate, y_validate.astype(None)),
                        epochs = 500,
                        batch_size = 150,
                        callbacks=[early_stopping_monitor1],
                        verbose=1)

    del y_train, y_validate

# Testing
def cls_predict(pred, normalize=True, sample_weight=None):
    s_mean = np.mean(pred, axis=0)
    return(list(s_mean))

def final_model(filename, segmentSize):
    print('Extracting features based on trained model...... ')
    dataframe2 = pd.read_csv(filename, header=None)
    dataset2 = dataframe2.values
    overlap = 50
    X_test = dataset2[:,0]
    Y_test = dataset2[:,1:len(dataset2[0])]
    c_p = []
    for tag, row in enumerate(X_test):
        pos = math.ceil(len(row) / overlap)
        if(pos < math.ceil(segmentSize/ overlap)):
            pos = math.ceil(segmentSize/ overlap)
        segment = [ ]
        for itr in range(pos - math.ceil(segmentSize/overlap) + 1):
            init = itr * overlap
            segment.append(row[init : init + segmentSize])
        seg_nGram = nGram(segment, chunkSize, dict_Prop)
        test_seg = sequence.pad_sequences(seg_nGram, maxlen=max_seq_len)
        preds = model.predict(test_seg, verbose = 0)
        c_p.append(cls_predict(preds))
    c_p = np.array(c_p)
    return c_p, Y_test

X_test_new, Y_test_new = final_model("mf/testData1.csv", segmentSize)
print(X_test_new.shape, Y_test_new.shape)
Y_test_new = np.array(Y_test_new).astype(None)

def test_fun(X_test_new, Y_test_new):
    fmax, tmax = 0.0, 0.0
    precisions, recalls = [], []
    for t in range(0, 101, 1):
        #test_preds = model1.predict(X_test_new)
        test_preds = np.copy(X_test_new)

        threshold = t / 100.0
        #print("THRESHOLD IS =====> ", threshold)
        test_preds[test_preds>=threshold] = int(1)
        test_preds[test_preds<threshold] = int(0)

        rec = recall(Y_test_new, test_preds)
        pre = precision(Y_test_new, test_preds)
        if math.isnan(pre):
            pre = 1.0
        recalls.append(rec)
        precisions.append(pre)

        f = 2 * pre * rec / (pre + rec)
        
        if fmax < f:
            fmax = f
            tmax = threshold

    test_preds = np.copy(X_test_new)
    print("THRESHOLD IS =====> ", tmax)
    test_preds[test_preds>=tmax] = int(1)
    test_preds[test_preds<tmax] = int(0)

    rec = recall(Y_test_new, test_preds)
    pre = precision(Y_test_new, test_preds)

    f = 2 * pre * rec / (pre + rec)
    print('Recall: {0}'.format(rec*100), '     Precision: {0}'.format(pre*100), '     F1-score1: {0}'.format(f*100))

    # COMPUTE AUPR
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    #aupr = np.trapezoid(precisions, recalls)   # new version.
    aupr = np.trapz(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')

    return tmax

th_set = test_fun(X_test_new, Y_test_new)

# Computing AUROC
# Flatten if multilabel
y_true = Y_test_new.flatten()
y_scores = X_test_new.flatten()   # RAW predictions (no threshold)

# AUROC
auroc = roc_auc_score(y_true, y_scores)
print(f'AUROC: {auroc:.3f}')
