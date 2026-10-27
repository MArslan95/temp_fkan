import tensorflow as tf

# **Spatial-Spectral Mamba Class**

class SpectralSpatialTokenGeneration1(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialTokenGeneration1, self).__init__(**kwargs)
        self.spatial_tokens = tf.keras.layers.Dense(out_channels)
        self.spectral_tokens = tf.keras.layers.Dense(out_channels)

    def call(self, x):
        B, H, W, C = x.shape
        # Use tf.shape to handle dynamic batch dimension
        spatial_tokens = self.spatial_tokens(tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [tf.shape(x)[0], H * W, C]))
        spectral_tokens = self.spectral_tokens(tf.reshape(tf.transpose(x, [0, 1, 2, 3]), [tf.shape(x)[0], H * W, C]))
        return spatial_tokens, spectral_tokens

class SpectralSpatialFeatureEnhancement1(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):  # Add 'self'
        super(SpectralSpatialFeatureEnhancement1, self).__init__(**kwargs)
        self.spatial_gate = tf.keras.Sequential([
            tf.keras.layers.Dense(out_channels),
            tf.keras.layers.Activation('sigmoid'),
            # Reshape the output to match spatial_tokens
            tf.keras.layers.Reshape((1, out_channels))
        ])
        self.spectral_gate = tf.keras.Sequential([
            tf.keras.layers.Dense(out_channels),
            tf.keras.layers.Activation('sigmoid'),
            # Reshape the output to match spectral_tokens
            tf.keras.layers.Reshape((1, out_channels))
        ])

    def call(self, spatial_tokens, spectral_tokens, center_tokens):
        spatial_enhanced = spatial_tokens * self.spatial_gate(center_tokens)
        spectral_enhanced = spectral_tokens * self.spectral_gate(center_tokens)
        return spatial_enhanced, spectral_enhanced

class StateSpaceModel1(tf.keras.layers.Layer):
    def __init__(self, state_dim, **kwargs):  # Add 'self'
        super(StateSpaceModel1, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.state_transition = tf.keras.layers.Dense(state_dim)
        self.state_update = tf.keras.layers.Dense(state_dim)

    def call(self, x):
        state = tf.zeros([tf.shape(x)[0], self.state_dim])
        for t in range(tf.shape(x)[1]):
            state = self.state_transition(state) + self.state_update(x[:, t, :])
        return state

class SSMambaModel(tf.keras.Model):
    def __init__(self, out_channels, num_heads, state_dim, Num_Classes, dropout=0.1, **kwargs):
        super(SSMambaModel, self).__init__(**kwargs)
        self.token_generation = SpectralSpatialTokenGeneration1(out_channels)
        self.feature_enhancement = SpectralSpatialFeatureEnhancement1(out_channels)
        self.state_space_model = StateSpaceModel1(state_dim)
        self.Dense = tf.keras.layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.classifier = tf.keras.layers.Dense(Num_Classes, activation='softmax')

    def call(self, x):
        spatial_tokens, spectral_tokens = self.token_generation(x)
        center_tokens = spatial_tokens[:, x.shape[1] // 2, :]
        spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
        Combined_Tokens = tf.concat([spatial_enhanced, spectral_enhanced], axis=-1)
        state_output = self.state_space_model(Combined_Tokens)
        output = self.classifier(state_output)
        return output

def SSMamba(Tr, batch_size, Num_Classes):
    model = SSMambaModel(out_channels=64, num_heads=4, state_dim=128, Num_Classes=Num_Classes, dropout=0.1)
    _ = model(Tr[:batch_size])
    return model