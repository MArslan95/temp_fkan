"""Preserved FKANMamba and declared ablations; no training side effects.

Selective block follows Mamba-1 equations and uses a sequential TensorFlow scan.
It is not the optimized CUDA implementation; runtime claims must say so.
References: state-spaces/mamba, modules/mamba_simple.py and
ops/selective_scan_interface.py (accessed 2026-10-25).
"""
import math
import numpy as np
import tensorflow as tf
from . import legacy_model as legacy
from .protocol import VARIANTS


def selective_scan(u, delta, a, b, c, d):
    u, delta, a, b, c, d = [tf.convert_to_tensor(v, dtype=tf.float32)
                             for v in (u, delta, a, b, c, d)]
    initial = tf.zeros((tf.shape(u)[0], tf.shape(u)[2], tf.shape(a)[1]))
    def step(state, inputs):
        ut, dt, bt = inputs
        return (tf.exp(dt[:, :, None]*a)*state
                + dt[:, :, None]*bt[:, None, :]*ut[:, :, None])
    states = tf.scan(step, (tf.transpose(u, (1,0,2)), tf.transpose(delta, (1,0,2)),
                            tf.transpose(b, (1,0,2))), initializer=initial)
    return tf.einsum('lbdn,bln->bld', states, c) + u*d


class SelectiveMamba(tf.keras.layers.Layer):
    """Mamba-1 block, d_model=64, d_state=64, d_conv=4, expand=2.

    Exposes input-dependent B/C/delta and a stable negative diagonal A.
    No claim of official-kernel equivalence until external parity is checked.
    """
    def __init__(self, width=64, state_dim=64):
        super().__init__()
        self.inner = 2*width
        self.state_dim = state_dim
        self.rank = math.ceil(width/16)
        self.in_proj = tf.keras.layers.Dense(2*self.inner, use_bias=False)
        self.conv_kernel = self.add_weight('conv_kernel', (4, 1, self.inner, 1))
        self.conv_bias = self.add_weight('conv_bias', (self.inner,), initializer='zeros')
        self.x_proj = tf.keras.layers.Dense(self.rank+2*state_dim, use_bias=False)
        dt = np.exp(np.random.uniform(np.log(.001), np.log(.1), self.inner))
        bias = dt+np.log(-np.expm1(-dt))
        self.dt_proj = tf.keras.layers.Dense(
            self.inner, kernel_initializer=tf.keras.initializers.RandomUniform(
                -self.rank**-.5, self.rank**-.5),
            bias_initializer=tf.keras.initializers.Constant(bias))
        self.a_log = self.add_weight('A_log', (self.inner, state_dim),
            initializer=tf.keras.initializers.Constant(
                np.tile(np.log(np.arange(1, state_dim+1)), (self.inner, 1))))
        self.d = self.add_weight('D', (self.inner,), initializer='ones')
        self.out_proj = tf.keras.layers.Dense(width, use_bias=False)

    def call(self, x):
        u, z = tf.split(self.in_proj(x), 2, axis=-1)
        u = tf.pad(u[:, :, None, :], ((0,0),(3,0),(0,0),(0,0)))
        u = tf.nn.depthwise_conv2d(u, self.conv_kernel, strides=(1,1,1,1), padding='VALID')
        u = tf.nn.silu(u[:, :, 0, :]+self.conv_bias)
        dt, b, c = tf.split(self.x_proj(u), [self.rank, self.state_dim, self.state_dim], -1)
        y = selective_scan(u, tf.nn.softplus(self.dt_proj(dt)),
                           -tf.exp(self.a_log), b, c, self.d)
        return self.out_proj(y*tf.nn.silu(z))[:, -1, :]


def projection(kind):
    layers = [tf.keras.layers.LayerNormalization(axis=-1)]
    if kind == 'mlp':
        layers += [tf.keras.layers.Dense(128, activation=tf.nn.gelu)]
    layers += [tf.keras.layers.Dense(32)]
    return tf.keras.Sequential(layers)


class Ablation(legacy.SSFKSMambaModel):
    def __init__(self, variant):
        super().__init__(32, state_dim=64, num_heads=8, num_grids=8)
        self.variant = variant
        if variant in ('linear_qkv', 'mlp_qkv'):
            kind = 'mlp' if variant == 'mlp_qkv' else 'linear'
            self.fssa.query_kan = projection(kind)
            self.fssa.key_kan = projection(kind)
            self.fssa.value_kan = projection(kind)
        if variant == 'mlp_gates':
            self.feb.spatial_gate = tf.keras.Sequential([projection('mlp'),
                                                        tf.keras.layers.Activation('sigmoid')])
            self.feb.spectral_gate = tf.keras.Sequential([projection('mlp'),
                                                         tf.keras.layers.Activation('sigmoid')])
        if variant == 'without_feb':
            self.feb = None
        if variant == 'without_fssa':
            self.fssa = None
        if variant == 'mean_pool':
            self.state_space_model = None
        if variant == 'selective_mamba':
            self.state_space_model = SelectiveMamba()

    def call(self, x, training=False):
        spatial, spectral = self.token_generation(x)
        representative = spatial[:, x.shape[1]//2, :]
        if self.feb is not None:
            spatial, spectral = self.feb(spatial, spectral, representative)
        if self.fssa is not None:
            spatial = self.fssa(spatial, spatial, spatial, training=training)
            spectral = self.fssa(spectral, spectral, spectral, training=training)
        fused = tf.concat([spatial, spectral], axis=-1)
        state = (tf.reduce_mean(fused, axis=1) if self.state_space_model is None
                 else self.state_space_model(fused))
        return self.classifier(state)


def build_model(method, classes):
    legacy.Num_Classes = classes
    if method == 'FKANMamba':
        return legacy.SSFKSMambaModel(32, state_dim=64, num_heads=8, num_grids=8)
    if method in VARIANTS:
        return Ablation(method)
    raise ValueError(f'{method} requires an authenticated baseline adapter; see adapters.json')
