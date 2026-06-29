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

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, seq_len, embed_dim, w_size, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.w_size = w_size
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def create_window_mask(self, seq_len, w_size):
        seq_len = 48
        mat = np.ones((seq_len, seq_len))
        for index in range(seq_len):
            for j in range(max(0, index - w_size), min(index + w_size + 1, seq_len)):
                mat[index][j] = 0
        tensor = tf.convert_to_tensor(mat)
        return tf.cast(tensor, tf.bool)

    def attention(self, query, key, value, mask):
        score = tf.matmul(query, key, transpose_b=True)                         # Calculate attention.
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # Implement Masking.
        if self.w_size is not None and mask is not None:
            win_mask = self.create_window_mask(self.seq_len, self.w_size)        # Compute window mask.
            int_mask = tf.math.logical_or(tf.cast(mask[0], tf.bool), win_mask)   # Combine pad mask and window mask.
            int_mask = tf.math.logical_or(tf.cast(mask[1], tf.bool), tf.cast(int_mask, tf.bool))
            final_mask = tf.cast(int_mask, tf.float32)
            scaled_score += (final_mask * -1e5)
        elif mask is not None:                                                     # add the mask to the scaled tensor.
            final_mask = mask[0]
            scaled_score += (final_mask * -1e5)                                    # mask: Float tensor (..., seq_len_q, seq_len_k).

        weights = tf.nn.softmax(scaled_score, axis=-1)
        max_wt = tf.reduce_max(weights, axis = -1)
        scaled_weights = tf.divide(weights, max_wt[:,:,:,tf.newaxis])             # Scaled SOFTMAX
        reverse_final_mask = 1.0 - final_mask
        scaled_weights = tf.multiply(scaled_weights, reverse_final_mask) + 0.001  # Re-apply mask
        output = tf.matmul(scaled_weights, value)
        return output, scaled_weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)                                        # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)                                            # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)                                        # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(query, batch_size)                          # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)                              # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)                          # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])                             # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))         # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)                                      # (batch_size, seq_len, embed_dim)
        return output, weights

class TransformerBlock(layers.Layer):
    def __init__(self, seq_len, embed_dim, num_heads, ff_dim, w_size, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.w_size = w_size
        self.att = MultiHeadSelfAttention(seq_len, embed_dim, w_size, num_heads)          # Sub-layer 1
        self.ffn = keras.Sequential([layers.Dense(ff_dim, kernel_initializer='normal', activation="relu"),    # Sub-layer 2
                                     layers.Dense(embed_dim, kernel_initializer='normal'),])                  # Two linear transformations with ReLU activation in between.
        #self.ffn = keras.Sequential([KANLinear(ff_dim, base_activation="relu"),                               # Sub-layer 2
        #                             layers.Dense(embed_dim, kernel_initializer='normal'),])                  # Two linear transformations with ReLU activation in between.
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate=0.2)
        self.dropout2 = layers.Dropout(rate=0.2)

    def call(self, inputs, mask, training=True):                                           # Main transformer block
        attn_output, attn_wt = self.att(inputs, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attn_wt

    def get_config(self):
        config = super().get_config().copy()
        config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads, 'ff_dim': self.ff_dim,
                       'seq_len': self.seq_len, 'w_size': self.w_size})
        return config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.emded_dim = emded_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)
        self.dropout1 = layers.Dropout(rate=0.3)
        self.dropout2 = layers.Dropout(rate=0.3)

    def create_padding_mask(self, seq, batch_size):
        index = [[i] for i in range(0, max_seq_len - 1, 2)]     # Captures the alternate indices to implement sequence shorting.
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)    # add extra dimensions to add the padding to the attention logits.
        seq = tf.reshape(tf.gather(seq, indices = index, axis = -1), [batch_size, len(index)])     # Helps extract the alternate values from the seq
        return seq[:, tf.newaxis, tf.newaxis, :], seq[:, tf.newaxis, :, tf.newaxis]
        # return (batch_size, 1, 1, seq_len), (batch_size, 1, seq_len, 1)

    def call(self, x, training=True):
        maxlen = tf.shape(x)[-1]
        batch_size = tf.shape(x)[0]
        padding_mask = self.create_padding_mask(x, batch_size)

        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        positions = self.dropout1(positions, training=training)
        x = self.token_emb(x)
        x = self.dropout2(x, training=training)
        return x + positions, padding_mask

    def get_config(self):
        config = super().get_config().copy()
        config.update({'maxlen': self.maxlen, 'vocab_size': self.vocab_size,'emded_dim': self.emded_dim,})
        return config

def create_rec_model1(top_words, seq_len, o_dim):
    embedding_vecor_length = 80          # Embedding size for each token
    num_heads = 4                        # Number of attention heads
    ff_dim = 128                         # Hidden layer size in feed forward network inside transformer

    # N-GRAM INPUT REPRESENTATION
    inp_gram = layers.Input(shape=(seq_len,))
    embedding_layer = TokenAndPositionEmbedding(seq_len, top_words, embedding_vecor_length)
    x_gram, pad_mask = embedding_layer(inp_gram)
    x_gram = layers.Dropout(0.3)(x_gram)
    x = layers.AveragePooling1D(pool_size = 2, strides = 2)(x_gram)                             # Sequence Shorting

    transformer_block1 = TransformerBlock(seq_len, embedding_vecor_length, num_heads, ff_dim, 12)       # 13 - 1   ---------> 25 %
    x1, attn_score1 = transformer_block1(x, pad_mask)

    transformer_block2 = TransformerBlock(seq_len, embedding_vecor_length, num_heads, ff_dim, 24)       # 25 - 1   ---------> 50 %
    x2, attn_score2 = transformer_block2(x, pad_mask)

    transformer_block3 = TransformerBlock(seq_len, embedding_vecor_length, num_heads, ff_dim, seq_len)  # 50      ---------> Complete
    x3, attn_score3 = transformer_block3(x, pad_mask)

    # PHYSICOCHEMICAL INPUT REPRESENTATION
    inp_phy = layers.Input(shape=(segmentSize, 4), dtype=np.float32)
    x_phy = layers.Conv1D(filters = 128, kernel_size = 3, activation = 'relu', padding = "same")(inp_phy)
    x_phy = layers.Conv1D(filters = 64, kernel_size = 4, activation = 'relu')(x_phy)
    x_phy = layers.BatchNormalization()(x_phy)
    x_phy = layers.AveragePooling1D(pool_size = 2, strides = 2)(x_phy)                             # Sequence Shorting

    transformer_block_phy = TransformerBlock(seq_len, 64, num_heads, ff_dim, seq_len)  # 50      ---------> Complete
    x_phy, attn_score_phy = transformer_block_phy(x_phy, pad_mask)

    x = layers.Concatenate(axis = -1)([x1, x2, x3, x_phy])

    # compute importance for each step
    attention1 = layers.Dense(1, activation='tanh')(x)
    attention1 = layers.Flatten()(attention1)
    attention1 = layers.Activation('softmax')(attention1)

    attention2 = layers.Dense(1, activation='tanh')(x)
    attention2 = layers.Flatten()(attention2)
    attention2 = layers.Activation('softmax')(attention2)

    attention3 = layers.Dense(1, activation='tanh')(x)
    attention3 = layers.Flatten()(attention3)
    attention3 = layers.Activation('softmax')(attention3)

    attention = layers.Add()([attention1,attention2,attention3])
    attention = layers.RepeatVector(304)(attention)
    attention = layers.Permute([2, 1])(attention)

    sent_representation = layers.Multiply()([x, attention])

    # Sum over axis=1: (None, 304)
    def sum_over_time(x):
        return tf.reduce_sum(x, axis=1)

    sent_representation = layers.Lambda(sum_over_time, output_shape=(304,))(sent_representation)
    #sent_representation = layers.Lambda(lambda xin: keras.ops.sum(xin, axis=1))(sent_representation)

    x = layers.Dropout(0.3)(sent_representation)
    outputs = layers.Dense(o_dim, kernel_initializer='normal', activation='sigmoid')(x)

    r_model = keras.Model(inputs=[inp_gram, inp_phy], outputs=[outputs])

    adam = keras.optimizers.Adam(learning_rate=0.0008)
    r_model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=3.0), optimizer=adam, metrics=['binary_accuracy'])
    return r_model
