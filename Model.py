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

def pChemical(dataset, max_len):
    def normalize_attribute(aa_prop):
        max_item = max(list(aa_prop.values())) + 1
        min_item = min(list(aa_prop.values())) - 1
        for key in aa_prop.keys():
            aa_prop[key] = (aa_prop[key] - min_item) / (max_item - min_item)

    encoded_seg_data = []
    aa_side_chain_mass = {'A':89.079,  'R':174.188, 'N':132.104, 'D':133.089, 'C':121.145,
                          'Q':146.131, 'E':147.116, 'G':75.052,  'H':155.141, 'I':131.160,
                          'L':131.160, 'K':146.17,  'M':149.199, 'F':165.177, 'P':115.177,
                          'S':105.078, 'T':119.105, 'W':204.213, 'Y':181.176, 'V':117.133}

    aa_hydrophobic_value = {'A':1.8,  'R':-4.5,  'N':-3.5,  'D':-3.5,  'C':2.5,
                            'Q':-3.5, 'E':-3.5,  'G':-0.4,  'H':-3.2,  'I':4.5,
                            'L':3.8,  'K':-3.9,  'M':1.9,   'F':2.8,   'P':-1.6,
                            'S':-0.8, 'T':-0.7,  'W':-0.9,  'Y':-1.3,  'V':4.2}

    aa_hydrophilic_value = {'A':-0.5, 'R':3.0,   'N':0.2,   'D':3.0,   'C':-1.0,
                            'Q':0.2,  'E':3.0,   'G':0.0,   'H':-0.5,  'I':-1.8,
                            'L':-1.8, 'K':3.0,   'M':-1.3,  'F':-2.5,  'P':0.0,
                            'S':0.3,  'T':-0.4,  'W':-3.4,  'Y':-2.3,  'V':-1.5}

    aa_van_der_walls_value = {'A':67, 'R':148,   'N':96,   'D':91,   'C':86,
                            'Q':114,  'E':109,   'G':48,   'H':118,  'I':124,
                            'L':124, 'K':135,   'M':124,  'F':135,  'P':90,
                            'S':90,  'T':93,  'W':163,  'Y':141,  'V':105}

    normalize_attribute(aa_side_chain_mass)
    normalize_attribute(aa_hydrophobic_value)
    normalize_attribute(aa_hydrophilic_value)
    normalize_attribute(aa_van_der_walls_value)
    #print(aa_side_chain_mass, aa_hydrophobic_value, aa_hydrophilic_value)

    for j, row in enumerate(dataset):
        segMer = []
        for i in range(max_len - len(row)):
              encode = [0.0]*4
              segMer.append(encode)
        for i in range(len(row)):
            try:
              encode = [0.0]*4
              encode[0] = aa_side_chain_mass[row[i]]
              encode[1] = aa_hydrophobic_value[row[i]]
              encode[2] = aa_hydrophilic_value[row[i]]
              encode[3] = aa_van_der_walls_value[row[i]]
              segMer.append(encode)
            except:
              encode = [0.0]*4
              segMer.append(encode)
        encoded_seg_data.append((np.array(segMer)))
    return((np.array(encoded_seg_data)))

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
