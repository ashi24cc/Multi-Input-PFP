import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from Model import nGram, dictionary, pChemical, DC_CNN_Model
from sklearn.metrics import roc_auc_score
from Evaluate import recall, precision
import keras

train_path = "/data2017/bp/trainData.csv"
test_path = "/data2017/bp/testData.csv"

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

dataframe = pd.read_csv(train_path, header=None)
dataset = dataframe.values
print('Original Dataset Size : %s' %len(dataset))
X = dataset[:,0]
Y = dataset[:,1:len(dataset[0])]
del dataframe, dataset
print(X.shape, Y.shape)

# SEGMENTATION
segmentSize = 100
nonOL = segmentSize - 50
SEG = str(segmentSize)

X_seg, Y_seg = segment(X, Y, segmentSize, nonOL)
nb_of_cls = len(Y_seg[0])

#Split the dataset
x_tr, x_val, y_tr, y_val = train_test_split(X_seg, Y_seg, test_size = 0.1, random_state = 42)

y_train = np.array(y_tr, dtype=None)
y_validate = np.array(y_val, dtype=None)
print(len(x_tr), len(x_val))
print(y_train.shape, y_validate.shape)

# CREATING DICTIONARY
chunkSize = 4
dict_Prop = dictionary(train_path, chunkSize)
max_seq_len = segmentSize - chunkSize + 1

#CREATING N-GRAM REPRESENTATION
x_train_gram = nGram(x_tr, chunkSize, dict_Prop)
x_validate_gram = nGram(x_val, chunkSize, dict_Prop)

# truncate and pad input sequences
x_train_gram = sequence.pad_sequences(x_train_gram, maxlen=max_seq_len)
x_validate_gram = sequence.pad_sequences(x_validate_gram, maxlen=max_seq_len)
print(x_train_gram.shape, x_validate_gram.shape)

#CREATING PHYSICOCHEMICAL REPRESENTATION
x_train_phy = pChemical(x_tr, segmentSize)
x_validate_phy = pChemical(x_val, segmentSize)
print(x_train_phy.shape, x_validate_phy.shape)
del x_tr, x_val

# Create & Compile the model
model_recc = create_rec_model1(len(dict_Prop), max_seq_len, nb_of_cls)
print(model_recc.summary())

# Train The Recc Model
early_stopping_monitor1 = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
history = model_recc.fit([x_train_gram, x_train_phy], y_train.astype(None),
          validation_data = ([x_validate_gram, x_validate_phy], y_validate.astype(None)),
          epochs = 1000,
          batch_size = 150,
          callbacks=[early_stopping_monitor1],
          verbose=1)

del y_train, y_validate

# Testing
def cls_predict(pred, normalize=True, sample_weight=None):
    s_mean = np.mean(pred, axis=0)
    return(list(s_mean))

def final_model(filename):
    print('Extracting features based on LSTM model...... ')
    dataframe2 = pd.read_csv(filename, header=None)
    dataset2 = dataframe2.values
    overlap = 50
    X_test = dataset2[:,0]
    Y_test = dataset2[:,1:len(dataset2[0])]

    min_pos = math.ceil(segmentSize / overlap)
    all_segments = []
    segment_counts = []   # keeps track of which segments belong to which sequence

    # --------- Segment generation ----------
    for row in X_test:
        pos = max(math.ceil(len(row) / overlap), min_pos)

        segments = []
        for itr in range(pos - min_pos + 1):
            init = itr * overlap
            segments.append(row[init:init + segmentSize])

        segment_counts.append(len(segments))
        all_segments.extend(segments)

    # --------- nGram + padding (ONCE) ----------
    seg_nGram = nGram(all_segments, chunkSize, dict_Prop)
    test_seg_gram = sequence.pad_sequences(seg_nGram, maxlen=max_seq_len)
    test_seg_phy = pChemical(all_segments, segmentSize)
    del all_segments

    # --------- SINGLE model.predict ----------
    preds = model_recc.predict([test_seg_gram, test_seg_phy], batch_size=256, verbose=0)
    del test_seg_gram, test_seg_phy

    # --------- Reconstruct per-sequence predictions ----------
    c_p = []
    start = 0
    for count in segment_counts:
        seq_preds = preds[start:start + count]
        c_p.append(cls_predict(seq_preds))
        start += count
        
    return np.array(c_p), Y_test

recc_preds, Y_test_new = final_model(test_path)
print(recc_preds.shape, Y_test_new.shape)
Y_test_new = np.array(Y_test_new).astype(None)

# ESM-2 BASED MODEL
import torch
import esm
from tqdm import tqdm

# Make (name, sequence) pairs for ESM input
sequences = [(f"seq_{i+1}", seq) for i, seq in enumerate(X)]

# ============================================================
# Load pretrained ESM2 model
# ============================================================
model_name = "esm2_t6_8M_UR50D"  # smallest and Colab-friendly
model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
batch_converter = alphabet.get_batch_converter()

print(f"Model loaded: {model_name} | Device: {device}")

# ============================================================
# Extract embeddings (mean-pooled per sequence)
# ============================================================
batch_size = 8  # increase if you have more GPU RAM
embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting ESM2 embeddings"):
        batch = sequences[i:i+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        out = model(batch_tokens, repr_layers=[model.num_layers])
        token_reps = out["representations"][model.num_layers]

        for j, (_, seq) in enumerate(batch):
            seq_len = len(seq)
            seq_emb = token_reps[j, 1:seq_len+1].mean(0).cpu().numpy()  # mean pooling
            embeddings.append(seq_emb)

embeddings = np.vstack(embeddings)
print("Embeddings shape:", embeddings.shape)

#Split the dataset
x_tr, x_val, y_tr, y_val = train_test_split(embeddings, Y, test_size = 0.1, random_state = 42)
y_train = np.array(y_tr, dtype=None)
y_validate = np.array(y_val, dtype=None)

print(len(x_tr), len(x_val))
print(x_tr.shape, y_train.shape, x_val.shape, y_validate.shape)

dim = x_tr.shape[1]
print(dim)

from keras import Model, Input, layers
from keras import optimizers
from keras.callbacks import EarlyStopping

inputs = layers.Input(shape=(dim,))

# Hidden layers
x = layers.Dense(1024, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(nb_of_cls, kernel_initializer='normal', activation='sigmoid')(x)        # Output layer

# Build model
model_plm = Model(inputs=inputs, outputs=outputs)
model_plm.compile(loss='binary_crossentropy',
               optimizer = optimizers.Adam(learning_rate=1e-3),
               metrics=['binary_accuracy'])
print(model_plm.summary())
early_stopping_monitor1 = EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       verbose=1)
history = model_plm.fit(x_tr, y_train.astype(None),
          validation_data=(x_val, y_validate.astype(None)),
          epochs = 100,
          batch_size = 150,
          callbacks=[early_stopping_monitor1],
          verbose=1)

dataframe = pd.read_csv(test_path, header = None)
dataset = dataframe.values
X_test = dataset[:,0]
test_actual = dataset[:,1:len(dataset[0])]
print('Original Dataset Size : %s' %len(X_test))
print(test_actual.shape)

# Make (name, sequence) pairs for ESM input
sequences = [(f"seq_{i+1}", seq) for i, seq in enumerate(X_test)]

# Extract embeddings (mean-pooled per sequence
batch_size = 1  # increase if you have more GPU RAM
test_embeddings = []

with torch.no_grad():
  for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting ESM2 embeddings"):
    batch = sequences[i:i+batch_size]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)
    out = model(batch_tokens, repr_layers=[model.num_layers])
    token_reps = out["representations"][model.num_layers]

    for j, (_, seq) in enumerate(batch):
        seq_len = len(seq)
        seq_emb = token_reps[j, 1:seq_len+1].mean(0).cpu().numpy()  # mean pooling
        test_embeddings.append(seq_emb)

test_embeddings = np.vstack(test_embeddings)
print("Embeddings shape:", test_embeddings.shape)

plm_preds = model_plm.predict(test_embeddings)

from matplotlib import pyplot as plt

# Testing
def test_fun(recc_preds, plm_preds, Y_test_new):
    fmax, tmax = 0.0, 0.0
    precisions, recalls = [], []
    for t in range(0, 101, 1):
        test_preds_recc = np.copy(recc_preds)
        test_preds_plm = np.copy(plm_preds)
        test_preds = np.maximum(test_preds_recc, test_preds_plm)

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
        #print('Recall: {0}'.format(rec*100), '     Precision: {0}'.format(pre*100), '     F1-score1: {0}'.format(f*100))

        if fmax < f:
            fmax = f
            tmax = threshold

    test_preds = np.maximum(np.copy(recc_preds), np.copy(plm_preds))
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
    aupr = np.trapezoid(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')

    return tmax

th_set = test_fun(recc_preds, plm_preds, Y_test_new)
