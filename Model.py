import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(7)

def dictionary(chunk_size):
    dataframe = pd.read_csv("mf/trainData.csv", header=None)
    dataset = dataframe.values
    del dataframe

    seq_dataset = dataset[:,0]
    print('Creating Dictionary:')
    dict = {}
    j = 0
    for row in seq_dataset:
        for i in range(len(row) - chunk_size + 1):
            key = row[i:i + chunk_size]
            if key not in dict:
                dict[key] = j
                j = j + 1
    del dataset, seq_dataset
    return(dict)

def nGram(dataset, chunk_size, dictI):
    dict1 = list()
    for j, row in enumerate(dataset):
        string = row
        dict2 = list()
        for i in range(len(string) - chunk_size + 1):
            try:
                dict2.append(dictI[string[i:i + chunk_size]])
            except:
                None
        dict1.append(dict2)
    return(dict1)

def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
    def f(input_):
        residual = input_
        layer_out = layers.Conv1D(filters=nb_filter, kernel_size=filter_length, dilation_rate=dilation,
                                  activation='linear', padding='same', use_bias=True)(input_)
        layer_out = layers.BatchNormalization(epsilon=1.1e-5)(layer_out)
        layer_out = layers.Activation("gelu")(layer_out)
        return layer_out
    return f

embed_dim = 64
ff_dim = 960

def DC_CNN_Model(top_words, seq_len, o_dim):
    f_num = 192
    f_size = [6,6,6,6,6]

    _input = layers.Input(shape=(seq_len,))
    emd = layers.Embedding(top_words, embed_dim, input_length = seq_len)(_input)
    drop1 = layers.Dropout(0.3)(emd)

    l1 = DC_CNN_Block(f_num, f_size[0], 1, 0.001)(drop1)
    l2 = DC_CNN_Block(f_num, f_size[1], 3, 0.001)(drop1)
    l3 = DC_CNN_Block(f_num, f_size[2], 5, 0.001)(drop1)
    l4 = DC_CNN_Block(f_num, f_size[3], 7, 0.001)(drop1)
    l5 = DC_CNN_Block(f_num, f_size[4], 9, 0.001)(drop1)

    x = layers.Concatenate(axis = -1)([l1, l2, l3, l4, l5])

    sent_representation = layers.GlobalAveragePooling1D()(x)
    #sent_representation = layers.Lambda(lambda xin: keras.ops.sum(xin, axis=1))(sent_representation)

    x = layers.Dropout(0.4)(sent_representation)
    _output = layers.Dense(o_dim, kernel_initializer='normal', activation='sigmoid', name='CLASSIFIER')(x)

    model = keras.Model(inputs=_input, outputs=_output)
    model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005),
                  metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
    return model
