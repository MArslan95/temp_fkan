import tensorflow as tf 
import numpy as np
import pywt
# **Spatial-Spectral Wavelet Mamba**

class SpectralSpatialTokenGeneration(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialTokenGeneration, self).__init__(**kwargs)
        self.spatial_tokens = tf.keras.layers.Dense(out_channels)
        self.spectral_tokens = tf.keras.layers.Dense(out_channels)

    def call(self, x):
        B, H, W, C = x.shape
        # Use tf.shape to handle dynamic batch dimension
        spatial_tokens = self.spatial_tokens(tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [tf.shape(x)[0], H * W, C]))
        spectral_tokens = self.spectral_tokens(tf.reshape(tf.transpose(x, [0, 1, 2, 3]), [tf.shape(x)[0], H * W, C]))
        return spatial_tokens,spectral_tokens

class SpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):  # Add 'self'
        super(SpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
        self.spatial_gate = tf.keras.Sequential([
            tf.keras.layers.Dense(out_channels),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Reshape((1, out_channels))
        ])
        self.spectral_gate = tf.keras.Sequential([
            tf.keras.layers.Dense(out_channels),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Reshape((1, out_channels))
        ])

    def call(self, spatial_tokens, spectral_tokens, center_tokens):
        spatial_enhanced = spatial_tokens * self.spatial_gate(center_tokens)
        spectral_enhanced = spectral_tokens * self.spectral_gate(center_tokens)
        return spatial_enhanced, spectral_enhanced

class StateSpaceMode(tf.keras.layers.Layer):
    def __init__(self, state_dim, **kwargs):
        super(StateSpaceMode, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.state_transition = tf.keras.layers.Dense(units=state_dim, activation="relu")
        self.state_update = tf.keras.layers.Dense(units=state_dim, activation="relu")

    def call(self, x):
        state = tf.zeros([tf.shape(x)[0], self.state_dim])
        for t in range(tf.shape(x)[1]):
            # Flatten the input if it has more than 2 dimensions
            input_t = tf.reshape(x[:, t, :], [tf.shape(x)[0], -1])
            state = self.state_transition(state) + self.state_update(input_t)
        return state

class SSMambaModel(tf.keras.Model):
    def __init__(self, out_channels, num_heads, state_dim, Num_Classes, dropout=0.1, **kwargs):
        super(SSMambaModel, self).__init__(**kwargs)
        self.token_generation = SpectralSpatialTokenGeneration(out_channels)
        self.feature_enhancement = SpectralSpatialFeatureEnhancement(out_channels)
        self.state_space_model = StateSpaceMode(state_dim)
        self.dense = tf.keras.layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.classifier = tf.keras.layers.Dense(Num_Classes, activation='softmax')

    # def apply_wavelet_transform(self, image):
    #     """Apply a wavelet transform to the input tensor."""
    #     # Ensure the input is in the form expected by pywt
    #     image_np = tf.make_ndarray(tf.make_tensor_proto(image))
    #     """wavelet = 'haar', 'db1', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'"""
    #     # Apply wavelet transform
    #     coeffs = pywt.dwt2(image_np, 'coif1')
    #     cA, (cH, cV, cD) = coeffs
    #     # Combine coefficients (this example uses a simple concatenation)
    #     transformed = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])
    #     # Reshape to maintain 3D shape
    #     transformed_shape = (image_np.shape[0], image_np.shape[1], -1)
    #     transformed = transformed.reshape(transformed_shape)
    #     return tf.convert_to_tensor(transformed, dtype=tf.float32)
    def apply_wavelet_transform(self, image, wavelet_name='bior1.3'):
        """'haar', 'db1', 'db2', 'coif1', 'bior1.3'"""
        """Apply a wavelet transform to the input tensor."""
        image_np = tf.make_ndarray(tf.make_tensor_proto(image))
        if wavelet_name not in pywt.wavelist(kind='discrete'):
            raise ValueError(f"The wavelet {wavelet_name} is not supported for discrete wavelet transform.")
        coeffs = pywt.dwt2(image_np, wavelet_name)
        cA, (cH, cV, cD) = coeffs
        transformed = np.stack([cA, cH, cV, cD], axis=-1)
        return tf.convert_to_tensor(transformed, dtype=tf.float32)
    
    def call(self, x):
        spatial_tokens, spectral_tokens = self.token_generation(x)
        center_tokens = spatial_tokens[:, x.shape[1] // 2, :]
        spatial_tokens, spectral_tokens = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
        spatial_tokens = tf.map_fn(lambda t: tf.numpy_function(self.apply_wavelet_transform, [t], tf.float32), spatial_tokens, dtype=tf.float32)
        spectral_tokens = tf.map_fn(lambda t: tf.numpy_function(self.apply_wavelet_transform, [t], tf.float32), spectral_tokens, dtype=tf.float32)
        Combined_Tokens = tf.concat([spatial_tokens, spectral_tokens], axis=-1)
        state_output = self.state_space_model(Combined_Tokens)
        output = self.classifier(state_output)
        return output

def WaveMamba(Tr, batch_size, Num_Classes):
    model = SSMambaModel(out_channels=64, num_heads=4, state_dim=128, Num_Classes=Num_Classes, dropout=0.1)
    sample_input = Tr[:batch_size]
    _ = model(sample_input)
    return model