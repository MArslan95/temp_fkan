import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Concatenate, LayerNormalization, Conv2D, BatchNormalization, Layer, DepthwiseConv2D
from tensorflow.keras.models import Sequential, Model
class SpectralSpatialTokenGeneration(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialTokenGeneration, self).__init__(**kwargs)
        self.spatial_tokens = Dense(out_channels)
        self.spectral_tokens = Dense(out_channels)
    def call(self, x):
        B, H, W, C = x.shape
        spatial_tokens = self.spatial_tokens(tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [tf.shape(x)[0], H * W, C]))
        spectral_tokens = self.spectral_tokens(tf.reshape(tf.transpose(x, [0, 1, 2, 3]), [tf.shape(x)[0], H * W, C]))
        return spatial_tokens, spectral_tokens

class DiMA(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, memory_dim, dropout_rate=0.1, **kwargs):
        super(DiMA, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.memory_dim = memory_dim
        self.depth = d_model // num_heads
        
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        
        self.k_mem = Dense(self.depth)
        self.v_mem = Dense(self.depth)


        self.concat_project = Dense(d_model)
        self.dropout = Dropout(dropout_rate)
        
        # Learnable memory matrix
        self.memory = self.add_weight(
            shape=(1, num_heads, memory_dim, self.depth),
            initializer="glorot_uniform",
            trainable=True,
            name="memory_matrix"
        )

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def differential_attention(self, scores):
        diff_scores = scores[:, :, 1:] - scores[:, :, :-1]
        diff_scores = tf.pad(diff_scores, [[0, 0], [0, 0], [1, 0], [0, 0]])
        return diff_scores

    def call(self, q, k, v, training=False):
        batch_size = tf.shape(q)[0]

        q_proj = self.split_heads(self.wq(q), batch_size)
        k_proj = self.split_heads(self.wk(k), batch_size)
        v_proj = self.split_heads(self.wv(v), batch_size)

        # Standard attention
        scores = tf.matmul(q_proj, k_proj, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        diff_scores = self.differential_attention(scores)
        scores += diff_scores
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_output = tf.matmul(self.dropout(attn_weights, training=training), v_proj)

        # Memory-based attention
        k_mem_proj = self.k_mem(self.memory)  # [1, num_heads, memory_dim, depth]
        v_mem_proj = self.v_mem(self.memory)
        mem_scores = tf.matmul(q_proj, k_mem_proj, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        mem_weights = tf.nn.softmax(mem_scores, axis=-1)
        mem_output = tf.matmul(mem_weights, v_mem_proj)

        # Concatenate both outputs
        combined = tf.concat([attn_output, mem_output], axis=-1)
        combined = tf.transpose(combined, perm=[0, 2, 1, 3])  # [B, seq, H, 2*D]
        combined = tf.reshape(combined, [batch_size, -1, 2 * self.depth * self.num_heads])
        output = self.concat_project(combined)
        return output

class SpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
        self.spatial_gate = Sequential([
            Dense(out_channels),
            Activation('sigmoid'),
            Reshape((1, out_channels))
        ])
        self.spectral_gate = Sequential([
            Dense(out_channels),
            Activation('sigmoid'),
            Reshape((1, out_channels))
        ])
    def call(self, spatial_tokens, spectral_tokens, center_tokens):
        spatial_enhanced = spatial_tokens * self.spatial_gate(center_tokens)
        spectral_enhanced = spectral_tokens * self.spectral_gate(center_tokens)
        return spatial_enhanced, spectral_enhanced

class StateSpaceModel(tf.keras.layers.Layer):
    def __init__(self, state_dim, **kwargs):
        super(StateSpaceModel, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.state_transition = Dense(state_dim)
        self.state_update = Dense(state_dim)
    def call(self, x):
        state = tf.zeros([tf.shape(x)[0], self.state_dim])
        for t in range(tf.shape(x)[1]):
            state = self.state_transition(state) + self.state_update(x[:, t, :])
        return state

class SSMambaModel(tf.keras.Model):
    def __init__(self, out_channels, num_heads, state_dim, Num_Classes,dropout=0.1, **kwargs):
        super(SSMambaModel, self).__init__(**kwargs)
        self.pre_feature = tf.keras.Sequential([
    Conv2D(16, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(32, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization()
])

        self.token_generation = SpectralSpatialTokenGeneration(out_channels)
        self.multi_head_attention = DiMA(num_heads=num_heads, d_model=out_channels,memory_dim=16, dropout_rate=dropout)
        self.feature_enhancement = SpectralSpatialFeatureEnhancement(out_channels)
        self.state_space_model = StateSpaceModel(state_dim)
        self.dense = Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = Dropout(0.4)
        self.classifier = Dense(Num_Classes, activation='softmax')
    def call(self, x):
        x = self.pre_feature(x)

        spatial_tokens, spectral_tokens = self.token_generation(x)
        # center_tokens = spatial_tokens[:, x.shape[1] // 2, :]
        center_tokens = tf.reduce_mean(spatial_tokens, axis=1, keepdims=True)

        spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
        attention_output = self.multi_head_attention(spatial_enhanced, spectral_enhanced, spectral_enhanced)

        state_output = self.state_space_model(attention_output)
        output = self.classifier(state_output)
        return output
    
def DiMAMamba(Tr, batch_size, Num_Classes):
    model = SSMambaModel(out_channels=64, num_heads=4, state_dim=128, Num_Classes=Num_Classes, dropout=0.1)
    _ = model(Tr[:batch_size])
    return model