import pywt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
# **Wavelet Transformation**

def wavelet_transform_image(image):
  # Define the wavelet and level of decomposition
  wavelet = 'haar'
  level = 3
  # Apply wavelet transform to each image in the 4D array
  wavelet_transformed_image = np.zeros_like(image)
  for i in range(image.shape[0]):
    for j in range(image.shape[3]):
      coeffs = pywt.wavedec2(image[i, :, :, j], wavelet, level=level)
      wavelet_transformed_image[i, :, :, j] = pywt.waverec2(coeffs, wavelet)
  return wavelet_transformed_image

# **Transformer Arch.**

def get_positional_encoding(max_len, d_emb):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_emb)[np.newaxis, :]
    angles = pos / np.power(10000, 2 * i / d_emb)
    positional_encoding = np.zeros((max_len, d_emb))
    positional_encoding[:, ::2] = np.sin(angles[:, ::2])
    positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return positional_encoding[np.newaxis, ...]

class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_heads, transformer_layer_depth, d_model, dff, dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.transformer_layer_depth = transformer_layer_depth
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.layernorm_eps = layernorm_eps

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

    def call(self, inputs, training):
        x = inputs
        for i in range(self.transformer_layer_depth):
            # MultiHead Attention
            attn_output = self.mha(x, x)
            attn_output = self.dropout1(attn_output, training=training)
            attn_output = self.layernorm1(x + attn_output)
            # Feed Forward Network
            ffn_output = self.dense1(attn_output)
            ffn_output = self.dense2(ffn_output)
            ffn_output = self.dropout2(ffn_output, training=training)
            ffn_output = self.layernorm2(attn_output + ffn_output)
            x = ffn_output
        return x

## 3D Pyramid Model
def pyramid_3d_cnn(WS, k, num_pyramid_levels):
    input_shape = (WS, WS, k, 1)
    max_patch_size = min(input_shape[:3])
    patch_size = max_patch_size
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    # Create pyramid levels
    pyramid_outputs = []
    # Create pyramid levels
    for level in range(num_pyramid_levels):
        scale = 2 ** level
        scaled_input_shape = (input_shape[0] // scale, input_shape[1] // scale, input_shape[2] // scale, input_shape[3])
        # Convolutional layers for each pyramid level
        x = tf.keras.layers.Conv3D(32, kernel_size=(1, patch_size, patch_size), activation='relu', padding='same', input_shape=scaled_input_shape)(x)
        x = tf.keras.layers.Conv3D(64, kernel_size=(1, patch_size, patch_size), activation='relu', padding='same')(x)
        pyramid_outputs.append(x)

    # Merge the feature maps from pyramid levels
    merged = tf.keras.layers.concatenate(pyramid_outputs, axis=-1)
    return Model(inputs=inputs, outputs=merged)

def PyFormer(WS, k, num_classes):
    num_pyramid_levels=2
    num_layers = 2
    num_heads = 4
    mlp_dim = 64
    input_shape = (WS, WS, k, 1)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = pyramid_3d_cnn(WS, k, num_pyramid_levels)(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    for i in range(num_layers):
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=k)(x, x)
        x = tf.keras.layers.Add()([x, inputs])
        y = tf.keras.layers.Conv3D(filters=k, kernel_size=(1, WS, WS), activation='relu', padding='same')(x)
        y = tf.keras.layers.Conv3D(filters=2*mlp_dim, kernel_size=(1, WS, WS), activation='relu', padding='same')(y)
        x = tf.keras.layers.Add()([x, y])
    # Apply regularization techniques
    flatten_layer = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(flatten_layer)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

## Hybrid ViT
def Hybrid_ViT(WS, k, Num_Classes, num_transformer_layers=4, num_heads=8, dff=4*64, dropout_rate=0.1, layernorm_eps=1e-3):
    input_shape = (WS, WS, k)
    max_patch_size = min(input_shape[:3])
    patch_size = max_patch_size
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size) * (input_shape[2] // patch_size)
    embedding_dim = 64
    inputs = tf.keras.layers.Input(shape=input_shape, name='inputs')
    x = tf.expand_dims(inputs, axis=1) # add this line
    # Create a patch embedding layer
    patch_embed = tf.keras.layers.Conv3D(embedding_dim, kernel_size=(1, patch_size, patch_size), strides=(1, patch_size, patch_size), padding='valid', name='patch_embed')(x)
    cls_token = tf.Variable(tf.zeros((1, 1, 1, embedding_dim)), name='cls_token')
    # Add cls_token to patch embeddings
    patch_embed = tf.concat([tf.broadcast_to(cls_token, [tf.shape(patch_embed)[0], 1, 1, 1, embedding_dim]), patch_embed], axis=1)
    # Add positional encoding to patch embeddings
    pos_encoding = get_positional_encoding(num_patches+1, embedding_dim)
    patch_embed = patch_embed + pos_encoding
    # Create a Transformer encoder
    for i in range(num_transformer_layers):
        if i == 0:
            transformer_layer = Transformer(num_heads=num_heads, transformer_layer_depth=1, d_model=embedding_dim, dff=dff, name=f'transformer_layer_{i}')(patch_embed)
        else:
            transformer_layer = Transformer(num_heads=num_heads, transformer_layer_depth=1, d_model=embedding_dim, dff=dff, name=f'transformer_layer_{i}')(transformer_layer)

    # Use cross-attention between the patch embeddings and a separate set of learnable embeddings
    q = tf.keras.layers.Dense(embedding_dim)(transformer_layer[:, 0, 0, 0])
    k = tf.keras.layers.Dense(embedding_dim)(patch_embed[:, 1:, :, :, :])
    v = patch_embed[:, 1:, :, :, :]
    cross_attn_output = tf.matmul(q[:, tf.newaxis, :], k, transpose_b=True)
    cross_attn_output = cross_attn_output / tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))
    cross_attn_output = tf.nn.softmax(cross_attn_output, axis=-1)
    cross_attn_output = tf.matmul(cross_attn_output, v)
    # Concatenate cls_token and cross-attention output
    output_layer = tf.concat([transformer_layer[:, 0, 0, 0], cross_attn_output[:, 0, 0, 0]], axis=-1)
    # Add a few dense layers and a final softmax layer
    output_layer = tf.keras.layers.Dense(dff, activation='relu')(output_layer)
    output_layer = tf.keras.layers.Dense(dff, activation='relu')(output_layer)
    output_layer = tf.keras.layers.Dense(Num_Classes, activation='softmax', name='output_layer')(output_layer)
    # Define the model
    model = Model(inputs=inputs, outputs=output_layer, name='Hybrid_ViT')
    return model

# **Spatial-Spectral Transformer**

def SSFormer(WS, k, Num_Classes, num_transformer_layers=4, num_heads=8, dff=4*64, dropout_rate=0.1, layernorm_eps=1e-3):
    input_shape = (WS, WS, k)
    inputs = tf.keras.layers.Input(shape=input_shape, name='inputs')

    max_patch_size = min(input_shape[:3])
    patch_size = max_patch_size
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size) * (input_shape[2] // patch_size)
    embedding_dim = 128

    x = tf.expand_dims(inputs, axis=1) # add this line
    # Create a patch embedding layer
    patch_embed = tf.keras.layers.Conv3D(embedding_dim, kernel_size=(1, patch_size, patch_size),
                         padding='valid', name='patch_embed')(x)

    cls_token = tf.Variable(tf.zeros((1, 1, 1, embedding_dim)), name='cls_token')
    # Add cls_token to patch embeddings
    patch_embed = tf.concat([tf.broadcast_to(cls_token, [tf.shape(patch_embed)[0], 1, 1, 1, embedding_dim]), patch_embed], axis=1)
    # Add positional encoding to patch embeddings
    pos_encoding = get_positional_encoding(num_patches+1, embedding_dim)

    patch_embed = patch_embed + pos_encoding
    # Create a Transformer encoder
    for i in range(num_transformer_layers):
        if i == 0:
            transformer_layer = Transformer(num_heads=num_heads,
                                            transformer_layer_depth=1, d_model=embedding_dim,
                                            dff=dff, name=f'transformer_layer_{i}')(patch_embed)
        else:
            transformer_layer = Transformer(num_heads=num_heads, transformer_layer_depth=1,
                                            d_model=embedding_dim, dff=dff,
                                            name=f'transformer_layer_{i}')(transformer_layer)

    # Use cross-attention between the patch embeddings and a separate set of learnable embeddings
    q = tf.keras.layers.Dense(embedding_dim)(transformer_layer[:, 0, 0, 0])
    k = tf.keras.layers.Dense(embedding_dim)(patch_embed[:, 1:, :, :, :])
    v = patch_embed[:, 1:, :, :, :]
    cross_attn_output = tf.matmul(q[:, tf.newaxis, :], k, transpose_b=True)
    cross_attn_output = cross_attn_output / tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))
    cross_attn_output = tf.nn.softmax(cross_attn_output, axis=-1)
    cross_attn_output = tf.matmul(cross_attn_output, v)
    # Concatenate cls_token and cross-attention output
    output_layer = tf.concat([transformer_layer[:, 0, 0, 0], cross_attn_output[:, 0, 0, 0]], axis=-1)
    # Add a few dense layers and a final softmax layer
    output_layer1 = tf.keras.layers.Dense(dff, activation='relu')(output_layer)
    output_layer2 = tf.keras.layers.Dense(dff, activation='relu')(output_layer1)
    output_layer3 = tf.keras.layers.Dense(Num_Classes, activation='softmax', name='output_layer')(output_layer2)
    # Define the model
    model = Model(inputs=inputs, outputs=output_layer3, name='SSFormer')
    return model

# **WaveFormer**

def WaveFormer(WS, k, Num_Classes, num_transformer_layers=4, num_heads=8, dff=4*64, dropout_rate=0.1, layernorm_eps=1e-3):
    input_shape = (WS, WS, k)
    inputs = tf.keras.layers.Input(shape=input_shape, name='inputs')

    max_patch_size = min(input_shape[:3])
    patch_size = max_patch_size
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size) * (input_shape[2] // patch_size)
    embedding_dim = 128

    x = tf.expand_dims(inputs, axis=1) # add this line
    # Create a patch embedding layer
    patch_embed = tf.keras.layers.Conv3D(embedding_dim, kernel_size=(1, patch_size, patch_size),
                         padding='valid', name='patch_embed')(x)

    cls_token = tf.Variable(tf.zeros((1, 1, 1, embedding_dim)), name='cls_token')
    # Add cls_token to patch embeddings
    patch_embed = tf.concat([tf.broadcast_to(cls_token, [tf.shape(patch_embed)[0], 1, 1, 1, embedding_dim]), patch_embed], axis=1)
    # Add positional encoding to patch embeddings
    pos_encoding = get_positional_encoding(num_patches+1, embedding_dim)

    patch_embed = patch_embed + pos_encoding
    # Create a Transformer encoder
    for i in range(num_transformer_layers):
        if i == 0:
            transformer_layer = Transformer(num_heads=num_heads,
                                            transformer_layer_depth=1, d_model=embedding_dim,
                                            dff=dff, name=f'transformer_layer_{i}')(patch_embed)
        else:
            transformer_layer = Transformer(num_heads=num_heads, transformer_layer_depth=1,
                                            d_model=embedding_dim, dff=dff,
                                            name=f'transformer_layer_{i}')(transformer_layer)

    # Use cross-attention between the patch embeddings and a separate set of learnable embeddings
    q = tf.keras.layers.Dense(embedding_dim)(transformer_layer[:, 0, 0, 0])
    k = tf.keras.layers.Dense(embedding_dim)(patch_embed[:, 1:, :, :, :])
    v = patch_embed[:, 1:, :, :, :]
    cross_attn_output = tf.matmul(q[:, tf.newaxis, :], k, transpose_b=True)
    cross_attn_output = cross_attn_output / tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))
    cross_attn_output = tf.nn.softmax(cross_attn_output, axis=-1)
    cross_attn_output = tf.matmul(cross_attn_output, v)
    # Concatenate cls_token and cross-attention output
    output_layer = tf.concat([transformer_layer[:, 0, 0, 0], cross_attn_output[:, 0, 0, 0]], axis=-1)
    # Add a few dense layers and a final softmax layer
    output_layer1 = tf.keras.layers.Dense(dff, activation='relu')(output_layer)
    output_layer2 = tf.keras.layers.Dense(dff, activation='relu')(output_layer1)
    output_layer3 = tf.keras.layers.Dense(Num_Classes, activation='softmax', name='output_layer')(output_layer2)
    # Define the model
    model = Model(inputs=inputs, outputs=output_layer3, name='WaveFormer')
    return model