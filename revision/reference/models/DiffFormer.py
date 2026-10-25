import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.layers import (Conv3D, Dense, Dropout, LayerNormalization,
                          Flatten, Input, Reshape, concatenate, Reshape, ReLU)

class SWiGLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SWiGLU, self).__init__(**kwargs)

    def call(self, inputs):
        x, gate = tf.split(inputs, 2, axis=-1)
        return x * tf.sigmoid(gate) + x

def Positional_Encoding(max_len, d_emb):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_emb)[np.newaxis, :]
    angles = pos / np.power(10000, 2 * i / d_emb)
    positional_encoding = np.zeros((max_len, d_emb))
    positional_encoding[:, ::2] = np.sin(angles[:, ::2])
    positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return positional_encoding[np.newaxis, ...]

class DMHSA(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dropout_rate=0.1, **kwargs):
        super(DMHSA, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)
        self.dropout = Dropout(dropout_rate)
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def differential_attention(self, scores):
        diff_scores = scores[:, 1:] - scores[:, :-1]
        diff_scores = tf.pad(diff_scores, [[0, 0], [1, 0], [0, 0], [0, 0]])
        return diff_scores
    def call(self, q, k, v, training):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        scores = tf.matmul(q, k, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        diff_scores = self.differential_attention(scores)
        scores = scores + diff_scores
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.dense(output)

class SST(tf.keras.layers.Layer):
    def __init__(self, num_heads, transformer_layer_depth, d_model, dff, dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super(SST, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.transformer_layer_depth = transformer_layer_depth
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.layernorm_eps = layernorm_eps
        self.mha = DMHSA(num_heads=num_heads, d_model=d_model, dropout_rate=dropout_rate)
        self.dropout1 = Dropout(dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.dense1 = Dense(dff)
        self.relu1=ReLU()
        self.swiGLU1 = SWiGLU()
        self.dense2 = Dense(d_model)
        self.dropout2 = Dropout(dropout_rate)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
    def call(self, inputs, training):
        x = inputs
        for _ in range(self.transformer_layer_depth):
            attn_output = self.mha(x, x, x, training=training)
            attn_output = self.dropout1(attn_output, training=training)
            attn_output = self.layernorm1(x + attn_output)
            ffn_output = self.dense1(attn_output)
            ffn_output = self.relu1(ffn_output)
            ffn_output = self.dense2(ffn_output)
            ffn_output = self.dropout2(ffn_output, training=training)
            ffn_output = self.layernorm2(attn_output + ffn_output)
            x = ffn_output
        return x

def DSST(WS, k, Num_Classes, num_transformer_layers=2, num_heads=4, dff=4 * 64, dropout_rate=0.1, layernorm_eps=1e-3):
    input_shape = (WS, WS, k)
    max_patch_size = min(input_shape[:3])
    patch_size = max_patch_size
    num_patches = ((input_shape[0] // patch_size) * (input_shape[1] // patch_size) * (input_shape[2] // patch_size))
    embedding_dim = 64
    inputs = Input(shape=input_shape, name="inputs")
    x = tf.expand_dims(inputs, axis=1)
    patch_embed = Conv3D(embedding_dim, kernel_size=(1, patch_size, patch_size),
                         strides=(1, patch_size, patch_size), padding="valid", name="patch_embed")(x)
    patch_embed = tf.reshape(patch_embed, (-1, tf.reduce_prod(tf.shape(patch_embed)[1:-1]), embedding_dim)) 
    cls_token = tf.Variable(tf.zeros((1, 1, embedding_dim)), name="cls_token")
    patch_embed = tf.concat([tf.broadcast_to(cls_token, [tf.shape(patch_embed)[0], 1, embedding_dim]), patch_embed], axis=1)
    pos_encoding = Positional_Encoding(num_patches + 1, embedding_dim)
    pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
    pos_encoding = tf.reshape(pos_encoding, (1, -1, embedding_dim))
    patch_embed = patch_embed + pos_encoding
    transformer_layer = patch_embed
    for i in range(num_transformer_layers):
        transformer_layer = SST(
            num_heads=num_heads,
            transformer_layer_depth=2,
            d_model=embedding_dim,
            dff=dff,
            name=f"transformer_layer_{i}",
        )(transformer_layer)

    transformer_layer = tf.reshape(transformer_layer, (-1, tf.shape(transformer_layer)[1], embedding_dim))
    cls_token_output = transformer_layer[:, 0, :] 
    output_layer = Dense(32, kernel_regularizer=regularizers.l2(0.01))(cls_token_output)
    output_layer = Dense(Num_Classes, activation="softmax", name="output_layer")(output_layer)
    model = Model(inputs=inputs, outputs=output_layer, name="DSST")
    return model
