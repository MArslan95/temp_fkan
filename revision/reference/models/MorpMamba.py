import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Activation, Conv2D, DepthwiseConv2D
from tensorflow.keras.models import Sequential
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.query = Dense(self.all_head_size)
        self.key = Dense(self.all_head_size)
        self.value = Dense(self.all_head_size)
        self.dropout = Dropout(dropout)
    def call(self, query, key, value):
        batch_size = tf.shape(query)[0]
        query_proj = self.query(query)
        key_proj = self.key(key)
        value_proj = self.value(value)
        query_proj = tf.reshape(query_proj, [batch_size, -1, self.num_heads, self.head_dim])
        query_proj = tf.transpose(query_proj, [0, 2, 1, 3])
        key_proj = tf.reshape(key_proj, [batch_size, -1, self.num_heads, self.head_dim])
        key_proj = tf.transpose(key_proj, [0, 2, 1, 3])
        value_proj = tf.reshape(value_proj, [batch_size, -1, self.num_heads, self.head_dim])
        value_proj = tf.transpose(value_proj, [0, 2, 1, 3])
        attention_scores = tf.matmul(query_proj, key_proj, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, value_proj)
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, [batch_size, -1, self.all_head_size])
        attention_output = self.dropout(attention_output)
        return attention_output
  
## Spatial-Spectral Feature Enhancement
class SpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
        self.spatial_gate = Sequential([
            Dense(out_channels),
            Activation('sigmoid')
        ])
        self.spectral_gate = Sequential([
            Dense(out_channels),
            Activation('sigmoid')
        ])

    def call(self, spatial_tokens, spectral_tokens, center_tokens):
        # Spatial enhancement
        spatial_gate_output = self.spatial_gate(center_tokens)
        spatial_enhanced = spatial_tokens * spatial_gate_output[:, :, tf.newaxis]
        # Spectral enhancement
        spectral_gate_output = self.spectral_gate(center_tokens)
        spectral_gate_output = tf.expand_dims(spectral_gate_output, axis=-2)
        spectral_enhanced = spectral_tokens * spectral_gate_output
        return spatial_enhanced, spectral_enhanced
  
## Spatial-Spectral Morphology
class ErosionLayer(Layer):
    def __init__(self, kernel_size, **kwargs):
        super(ErosionLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
    def build(self, input_shape):
        channels = input_shape[-1]
        self.depthwise_conv = []
        for _ in range(channels):
            conv = DepthwiseConv2D(self.kernel_size, padding='same', use_bias=False)
            conv.build((None, None, None, 1))
            conv.set_weights([tf.ones((self.kernel_size, self.kernel_size, 1, 1))])
            self.depthwise_conv.append(conv)
    def call(self, x):
        eroded_channels = []
        for i in range(x.shape[-1]):
            channel = x[..., i:i+1]
            eroded_channel = -self.depthwise_conv[i](channel)
            eroded_channels.append(eroded_channel)
        return tf.concat(eroded_channels, axis=-1)

class DilationLayer(Layer):
    def __init__(self, kernel_size, **kwargs):
        super(DilationLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
    def build(self, input_shape):
        channels = input_shape[-1]
        self.depthwise_conv = []
        for _ in range(channels):
            conv = DepthwiseConv2D(self.kernel_size, padding='same', use_bias=False)
            conv.build((None, None, None, 1))
            conv.set_weights([tf.ones((self.kernel_size, self.kernel_size, 1, 1))])
            self.depthwise_conv.append(conv)
    def call(self, x):
        dilated_channels = []
        for i in range(x.shape[-1]):
            channel = x[..., i:i+1]
            dilated_channel = self.depthwise_conv[i](channel)
            dilated_channels.append(dilated_channel)
        return tf.concat(dilated_channels, axis=-1)

## Spatial-Spectral Token Generation for Spatial-Spectral Morphology
class SpectralSpatialTokenGeneration(Layer):
    def __init__(self, out_channels, kernel_size=5, **kwargs):
        super(SpectralSpatialTokenGeneration, self).__init__(**kwargs)
        self.erosion_layer_spatial = ErosionLayer(kernel_size)
        self.dilation_layer_spatial = DilationLayer(kernel_size)
        self.erosion_layer_spectral = ErosionLayer(kernel_size)
        self.dilation_layer_spectral = DilationLayer(kernel_size)
        self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size=1)
    def call(self, x):
        # Spatial morphology
        eroded_x_spatial = self.erosion_layer_spatial(x)
        dilated_x_spatial = self.dilation_layer_spatial(x)
        # Spectral morphology
        eroded_x_spectral = self.erosion_layer_spectral(tf.transpose(x, [0, 3, 1, 2]))
        dilated_x_spectral = self.dilation_layer_spectral(tf.transpose(x, [0, 3, 1, 2]))
        eroded_x_spectral = tf.transpose(eroded_x_spectral, [0, 2, 3, 1])
        dilated_x_spectral = tf.transpose(dilated_x_spectral, [0, 2, 3, 1])
        # Combine results
        combined_Spatial = self.conv(tf.concat([eroded_x_spatial, dilated_x_spatial], axis=-1))
        combined_Spectral = self.conv(tf.concat([eroded_x_spectral, dilated_x_spectral], axis=-1))
        return combined_Spatial, combined_Spectral

## State-Space Model
class StateSpaceModel(tf.keras.layers.Layer):
    def __init__(self, state_dim, **kwargs):  # Add 'self'
        super(StateSpaceModel, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.state_transition = Dense(state_dim)
        self.state_update = Dense(state_dim)
    def call(self, x):
        state = tf.zeros([tf.shape(x)[0], self.state_dim])
        for t in range(tf.shape(x)[1]):
            state = self.state_transition(state) + self.state_update(x[:, t, :])
        return state

## Spatial-Spectral Morpoholgy Mamba for HSIC
class SSMambaModel(tf.keras.Model):
    def __init__(self, out_channels, num_heads, state_dim,Num_Classes, dropout=0.1, **kwargs):
        super(SSMambaModel, self).__init__(**kwargs)
        self.token_generation = SpectralSpatialTokenGeneration(out_channels)
        self.multi_head_attention = MultiHeadAttention(out_channels, num_heads, dropout)
        self.feature_enhancement = SpectralSpatialFeatureEnhancement(out_channels)
        self.state_space_model = StateSpaceModel(state_dim)
        self.dense = Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = Dropout(0.4)
        self.classifier = Dense(Num_Classes, activation='softmax')
    def call(self, x):
        spatial_tokens, spectral_tokens = self.token_generation(x)
        center_tokens = spatial_tokens[:, x.shape[1] // 2, :]
        spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
        attention_output = self.multi_head_attention(spatial_enhanced, spectral_enhanced, spectral_enhanced)
        state_output = self.state_space_model(attention_output)
        dense_output = self.dense(state_output)
        output = self.classifier(dense_output)
        return output
    
def MorpMamba(Tr, batch_size, Num_Classes):
    model = SSMambaModel(out_channels=64, num_heads=4, state_dim=128, Num_Classes=Num_Classes, dropout=0.1)
    _ = model(Tr[:batch_size])
    return model