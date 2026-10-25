"""Exact class bodies from the incoming initial_code side of frozen main.py.
Not evidence that this configuration generated every historical result.
See reference/manifest.json and revision_audit.md for provenance limitations.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
num_heads = 8
Num_Classes = 16
class SpectralSpatialTokenGeneration(tf.keras.layers.Layer):
    """
    token generation for spectral and spatial features
    fully connected layer for spectral and spatial features
    """

    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialTokenGeneration, self).__init__(**kwargs)
        self.spatial_tokens = Dense(out_channels)  # pytorch nn.Linear
        self.spectral_tokens = Dense(out_channels)

    def call(self, x):
        B, H, W, C = x.shape
        # Use tf.shape to handle dynamic batch dimension
        spatial_tokens = self.spatial_tokens( tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [tf.shape(x)[0], H * W, C]))
        spectral_tokens = self.spectral_tokens(tf.reshape(tf.transpose(x, [0, 1, 2, 3]), [tf.shape(x)[0], H * W, C]))
        return spatial_tokens, spectral_tokens

class SplineLinear(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, init_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.init_scale = init_scale
        self.weight = self.add_weight(
            shape=(self.out_features, self.in_features),
            initializer=tf.keras.initializers.GlorotUniform(),  # tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=init_scale), #HeNormal(),
            trainable=True,
            regularizer=tf.keras.regularizers.l2(0.001),
        )

    def call(self, inputs):
        return tf.matmul(inputs, tf.transpose(self.weight))

class RadialBasisFunction(tf.keras.layers.Layer):
    def __init__(
        self, grid_min=-1.0, grid_max=1.0, num_grids=8, denominator=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
        self.grid = tf.Variable(
            initial_value=tf.linspace(grid_min, grid_max, num_grids),
            trainable=True,
            name="grid",
        )

    def call(self, inputs):
        expanded_inputs = tf.expand_dims(inputs, axis=-1)
        return tf.exp(-(((expanded_inputs - self.grid) / self.denominator) ** 2))

class FastKANLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        grid_min=-1.0,
        grid_max=1.0,
        num_grids=8,
        use_base_update=False,
        use_layernorm=True,
        base_activation=tf.nn.swish,
        spline_weight_init_scale=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_layernorm = use_layernorm
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        if use_layernorm:
            assert (
                input_dim > 1
            ), "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)
        else:
            self.layernorm = None
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(
            input_dim * num_grids, output_dim, init_scale=spline_weight_init_scale
        )
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            normalized_inputs = self.layernorm(inputs)
        else:
            normalized_inputs = inputs
        spline_basis = self.rbf(normalized_inputs)
        spline_basis_flat = tf.reshape(
            spline_basis, shape=(-1, spline_basis.shape[-2] * spline_basis.shape[-1])
        )
        ret = self.spline_linear(spline_basis_flat)
        # ret = tf.reshape(ret, shape=tf.shape(inputs))
        output_shape = tf.concat([tf.shape(inputs)[:-1], [self.output_dim]], axis=0)
        ret = tf.reshape(ret, shape=output_shape)
        if self.use_base_update:
            base = self.base_linear(self.base_activation(inputs))
            ret = ret + base
        return ret

class FastKANSpatialSpectralAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        num_heads,
        output_dim,
        grid_min=-1.0,
        grid_max=1.0,
        num_grids=8,
        use_base_update=False,
        base_activation=tf.nn.swish,
        spline_weight_init_scale=0.1,
        dropout_rate=0.7,
        output_activation=None,  # Optional activation for output
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.output_dim = output_dim
        self.all_head_size = self.num_heads * self.head_dim

        # Query, Key, and Value projections via FastKANLayer
        self.query_kan = FastKANLayer(
            input_dim=input_dim,
            output_dim=self.all_head_size,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=use_base_update,
            base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale,
        )
        self.key_kan = FastKANLayer(
            input_dim=input_dim,
            output_dim=self.all_head_size,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=use_base_update,
            base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale,
        )
        self.value_kan = FastKANLayer(
            input_dim=input_dim,
            output_dim=self.all_head_size,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=use_base_update,
            base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale,
        )
        # Final output projection
        self.output_projection = tf.keras.layers.Dense(
            units=self.output_dim
        )  # Better weight initialization
        # Optional output activation
        self.output_activation = (
            tf.keras.layers.Activation(output_activation) if output_activation else None
        )
        # Dropout for attention weights
        self.attention_dropout = tf.keras.layers.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension of the input into (num_heads, head_dim)
        Transpose the resulting tensor to (batch_size, num_heads, seq_len, head_dim)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        """
        Combines the (num_heads, head_dim) into a single dimension.
        Reshapes tensor to (batch_size, seq_len, output_dim).
        """
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (tf.shape(x)[0], -1, self.num_heads * self.head_dim))

    def call(self, queries, keys, values, training=False):
        batch_size = tf.shape(queries)[0]
        # Compute Query, Key, and Value embeddings using FastKANLayer
        Q = self.query_kan(queries)
        K = self.key_kan(keys)
        V = self.value_kan(values)
        # Split heads into (batch_size, num_heads, seq_len, head_dim)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        attention_scores = tf.matmul(
            Q, K, transpose_b=True
        )  # Shape: (batch_size, num_heads, seq_len, seq_len)
        d_k = tf.cast(self.head_dim, tf.float32)  # Head dimension
        attention_scores /= tf.math.sqrt(
            d_k
        )  # Scale by sqrt(head_dim) for numerical stability
        attention_weights = tf.nn.softmax(
            attention_scores, axis=-1
        )  # Softmax over sequence length
        # Dropout on attention weights
        attention_weights = self.attention_dropout(attention_weights, training=training)
        # Weighted sum of Value embeddings
        attention_output = tf.matmul(
            attention_weights, V
        )  # Shape: (batch_size, num_heads, seq_len, head_dim)
        # Combine heads back to (batch_size, seq_len, output_dim)
        attention_output = self.combine_heads(attention_output)
        # Final linear projection
        output = self.output_projection(attention_output)

        # Optional activation function
        if self.output_activation:
            output = self.output_activation(output)
        return output

class SpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialFeatureEnhancement, self).__init__(**kwargs)

        self.spatial_gate = Sequential(
            [
                FastKANLayer(
                    input_dim=out_channels,
                    output_dim=out_channels,
                    grid_min=-1.0,
                    grid_max=1.0,
                    num_grids=8,
                    use_base_update=False,
                    use_layernorm=True,
                ),
                Activation("sigmoid"),
            ]
        )

        self.spectral_gate = Sequential(
            [
                # FastKANLayer(input_dim=out_channels, output_dim=out_channels),
                FastKANLayer(
                    input_dim=out_channels,
                    output_dim=out_channels,
                    grid_min=-1.0,
                    grid_max=1.0,
                    num_grids=8,
                    use_base_update=False,
                    use_layernorm=True,
                ),
                Activation("sigmoid"),
            ]
        )

    def call(self, spatial_tokens, spectral_tokens, center_tokens):
        spatial_weights = self.spatial_gate(center_tokens)
        spectral_weights = self.spectral_gate(center_tokens)
        spatial_enhanced = spatial_tokens * tf.expand_dims(spatial_weights, axis=1)
        spectral_enhanced = spectral_tokens * tf.expand_dims(spectral_weights, axis=1)
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

class SSFKSMambaModel(tf.keras.Model):
    """
    token_generation -> feature_enhancement -> multi_head_attention -> concat -> state_space_model -> classifier(Dense)
    HSI 데이터를 받아서 token_generation을 통해 공간과 스펙트럴 토큰을 생성하고
    spatial_tokens의 중간 토큰을 gating으로 사용하여
    feature_enhancement를 통해 토큰을 강화하고
    공간과 스펙트럴 토큰 각각에 multi_head_attention을 적용하고
    어텐션 결과를 concat하여 state_space_model을 통해 시계열 모델링을 하고
    ssm의 결과를 classifier(Dense)에 통과시켜서 분류
    """

    def __init__(
        self, out_channels, state_dim=64, num_heads=num_heads, num_grids=8, **kwargs
    ):
        super(SSFKSMambaModel, self).__init__(**kwargs)
        # Tokenization
        self.token_generation = SpectralSpatialTokenGeneration(out_channels)
        # Multi-Head Attention
        self.fssa = FastKANSpatialSpectralAttention(
            input_dim=out_channels,
            output_dim=out_channels,
            num_heads=num_heads,
            grid_min=-1.0,
            grid_max=1.0,
            num_grids=num_grids,
            base_activation=tf.nn.swish,
        )
        # Feature Enhancement
        self.feb = SpectralSpatialFeatureEnhancement(out_channels)
        # State Space Model using Transformer
        self.state_space_model = StateSpaceModel(state_dim=state_dim)
        # Dense Layers and Classification
        self.dense = tf.keras.layers.Dense(
            units=out_channels,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        )
        self.dropout = tf.keras.layers.Dropout(0.7)
        self.classifier = tf.keras.layers.Dense(Num_Classes, activation="softmax")

    # Forward pass
    def call(self, x, training=False):
        # Step 1: Tokenization
        spatial_tokens, spectral_tokens = self.token_generation(x)
        center_tokens = spatial_tokens[
            :, x.shape[1] // 2, :
        ]  # Center tokens for gating
        # Step 2: Feature Enhancement
        spatial_enhanced, spectral_enhanced = self.feb(
            spatial_tokens, spectral_tokens, center_tokens
        )
        # # Step 3: Apply FSS Attention
        spatial_attention = self.fssa(
            spatial_enhanced, spatial_enhanced, spatial_enhanced, training=training
        )
        spectral_attention = self.fssa(
            spectral_enhanced, spectral_enhanced, spectral_enhanced, training=training
        )
        # Step 4: Combine the outputs
        combined_output = tf.concat([spatial_attention, spectral_attention], axis=-1)
        # combined_output = tf.concat([spatial_enhanced, spectral_enhanced], axis=-1)
        # Step 5: State Space Modeling using Transformer
        state_output = self.state_space_model(combined_output)
        # Step 6: Dense layer and classification
        output = self.classifier(state_output)
        return output
