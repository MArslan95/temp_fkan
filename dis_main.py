import time
import json
import sys
import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.colors import ListedColormap
import scipy.io as sio
import seaborn as sns
from scipy.signal import medfilt
from sklearn.manifold import TSNE
import spectral
import spectral.io.envi as envi
from typing import List
from operator import truediv
import keras
from keras.losses import categorical_crossentropy
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.optimizers import legacy
from keras import backend as Kb
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Activation, Lambda, multiply, Dense)
from tensorflow.keras.optimizers import Adam
from utils import *

# Set up argument parser
parser = argparse.ArgumentParser(description="Script Description")
parser.add_argument("--dataset", "-d", type=str, choices=["LK", "HC", "UH13", "PU", "PC", "SA","HH","QUH","PI","TH"], default="PI", help="Dataset identifier")
parser.add_argument("--window_size", "-ws", type=int, default=4, help="Window size")
parser.add_argument("--dl_method", "-dlm", type=str, default='PCA', help="Dimension Reduction Method")
parser.add_argument("--test_ratio", "-te", type=float, default=0.50, help="Test ratio")
parser.add_argument("--train_ratio", "-tr", type=float, default=0.99, help="Train ratio")
parser.add_argument("--validation_ratio", "-vr", type=float, default=0.01, help="Validation ratio")
parser.add_argument("--number_of_bands", "-k", type=int, default=15, help="Number of bands principal components for dimensionality reduction")
parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs for training")
parser.add_argument("--batch_size", "-bs", type=int, default=56, help="Batch size for training")
args = parser.parse_args()

# Check GPU availability
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Global Parameters
HSID = args.dataset
DLM = args.dl_method
WS = args.window_size
trRatio = args.train_ratio
vrRatio = args.validation_ratio
teRatio = args.test_ratio
k = args.number_of_bands
num_heads = 8
out_channels = 32
epochs = args.epochs
batch_size = args.batch_size
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=1e-06)
output_dir = os.path.join(f"KanMamba_{num_heads}_chnal_{out_channels}_bs_{batch_size}/{HSID}/{WS}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU(s): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
# Initialize MirroredStrategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Adjust batch size for multi-GPU
global_batch_size = batch_size * strategy.num_replicas_in_sync
print(f"Global batch size: {global_batch_size}")

# DataSet Loader Function
base_dir = '../hsi_datasets'
def LoadHSIData(method):
    data_path = os.path.join(os.getcwd(), f'{base_dir}')
    if method == 'LK':
        HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt'))['WHU_Hi_LongKou_gt']
        Num_Classes = 9
        target_names = ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean', 'Narrow-leaf soybean', 'Rice', 'Water', 'Roads and houses', 'Mixed weed']
    elif method == 'HH':
        HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
        GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_gt'))['WHU_Hi_HongHu_gt']
        Num_Classes = 22
        target_names = ['Red roof', 'Road', 'Bare soil', 'Cotton', 'Cotton firewood', 'Rape', 'Chinese cabbage', 'Pakchoi', 'Cabbage', 'Tuber mustard', 'Brassica parachinensis', 'Brassica chinensis', 'Small Brassica chinensis', 'Lactuca sativa', 'Celtuce', 'Film covered lettuce', 'Romaine lettuce', 'Carrot', 'White radish', 'Garlic sprout', 'Broad bean', 'Tree']
    elif method == 'HC':
        HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
        GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_gt'))['WHU_Hi_HanChuan_gt']
        Num_Classes = 16
        target_names = ['Strawberry', 'Cowpea', 'Soybean', 'Sorghum', 'Water spinach', 'Watermelon', 'Greens', 'Trees', 'Grass', 'Red roof', 'Gray roof', 'Plastic', 'Bare soil', 'Road', 'Bright object', 'Water']
    elif method == 'BS':
        HSI = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
        GT = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
        Num_Classes = 14
        target_names = ['Water', 'Hippo Grass', 'Floodplain Grasses 1', 'Floodplain Grasses 2', 'Reeds 1', 'Riparian', 'Firescar 2', 'Island Interior', 'Woodlands', 'Acacia Shrublands', 'Acacia Grasslands', 'Short Mopane', 'Mixed Mopane', 'Exposed Soils']
    elif method == 'PU':
        HSI = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        GT = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        Num_Classes = 9
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted', 'Soil', 'Bitumen', 'Bricks', 'Shadows']
    elif method == 'PC':
        HSI = sio.loadmat(os.path.join(data_path, 'Pavia.mat'))['pavia']
        GT = sio.loadmat(os.path.join(data_path, 'Pavia_gt.mat'))['pavia_gt']
        Num_Classes = 9
        target_names = ['Water', 'Trees', 'Asphalt', 'Bricks', 'Bitumen', 'Tiles', 'Shadows', 'Meadows', 'Soil']
    elif method == 'SA':
        HSI = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        GT = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        Num_Classes = 16
        target_names = ['Weeds_1', 'Weeds_2', 'Fallow', 'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_Weeds', 'Lettuce_4wk', 'Lettuce_5wk', 'Lettuce_6wk', 'Lettuce_7wk', 'Vinyard_untrained', 'Vinyard_trellis']
    elif method == 'UH13':
        HSI = sio.loadmat(os.path.join(data_path, 'HU13.mat'))['HSI']
        GT = sio.loadmat(os.path.join(data_path, 'HU13_gt.mat'))['gt']
        Num_Classes = 15
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis Court', 'Running Track']
    elif method == 'QUH':
        HSI = sio.loadmat(os.path.join(data_path, 'QUH-Qingyun.mat'))['Chengqu']
        GT = sio.loadmat(os.path.join(data_path, 'QUH-Qingyun_GT.mat'))['ChengquGT']
        Num_Classes = 6
        target_names = ["Trees", "Concrete building", "Car", "Ironhide building", "Plastic playground", "Asphalt road"]
    elif method == 'PI':
        HSI = sio.loadmat(os.path.join(data_path, 'QUH-Pingan.mat'))['Haigang']
        GT = sio.loadmat(os.path.join(data_path, 'QUH-Pingan_GT.mat'))['HaigangGT']
        Num_Classes = 10
        target_names = ["Ship", "Seawater", "Trees", "Concrete structure building", "Floating pier", "Brick houses", "Steel houses", "Wharf construction land", "Car", "Road"]
    elif method == 'TH':
        HSI = sio.loadmat(os.path.join(data_path, 'QUH-Tangdaowan.mat'))['Tangdaowan']
        GT = sio.loadmat(os.path.join(data_path, 'QUH-Tangdaowan_GT.mat'))['TangdaowanGT']
        Num_Classes = 18
        target_names = ['Rubber track', 'Flagging', 'Sandy', 'Asphalt', 'Boardwalk', 'Rocky shallows', 'Grassland', 'Bulrush', 'Gravel road', 'Ligustrum vicaryi', 'Coniferous pine', 'Spiraea', 'Bare soil', 'Buxus sinica', 'Photinia serrulata', 'Populus', 'Ulmus pumila L', 'Seawater']
    else:
        print("Wrong Choice")
    return HSI, GT, Num_Classes, target_names

# Create Patches for 3D (Spatial-Spectral) Models
def ImageCubes(HSI, GT, WS=WS, removeZeroLabels=True):
    '''
    HSI 데이터를 Image Cubes로 변환하는 함수. Image Cubes는 3D 데이터로, 각 픽셀 주변의 윈도우 크기만큼의 데이터를 포함
    
    HSI: np.array : hyperspectral image
    GT: np.array : ground truth
    WS: int : window size
    removeZeroLabels: bool : remove zero labels
    '''
    num_rows, num_cols, num_bands = HSI.shape   # HSI data shape (width, height, bands)
    margin = int(WS / 2)                        # padding margin
    padded_data = np.pad(HSI, ((margin, margin), (margin, margin), (0, 0)), mode='constant')    # zero padding
    image_cubes = np.zeros((num_rows * num_cols, WS, WS, num_bands)) # array initialize with zeros for image cubes
    patchesLabels = np.zeros((num_rows * num_cols)) # array initialize with zeros for patches labels (number of pixels)
    patchIndex = 0  # patch index initialize with zero
    for r in range(margin, num_rows + margin):
        for c in range(margin, num_cols + margin):
            cube = padded_data[r - margin: r + margin, c - margin: c + margin, :]   #  Extract a window size cube centred on the current pixel
            image_cubes[patchIndex, :, :, :] = cube     # Assign the cube to the image cubes array
            patchesLabels[patchIndex] = GT[r-margin, c-margin]  # Assign the ground truth label to the patches labels array
            patchIndex = patchIndex + 1     # Increment the patch index
    if removeZeroLabels:    # Remove the zero labels
      image_cubes = image_cubes[patchesLabels>0,:,:,:]  # Remove the zero labels from the image cubes
      patchesLabels = patchesLabels[patchesLabels>0]    # Remove the zero labels from the patches labels
      patchesLabels -= 1    # Subtract 1 from the patches labels to shift the class labels
    return image_cubes, patchesLabels   # Return the image cubes and patches labels

# Load and preprocess data
HSI, GT, Num_Classes, target_names = LoadHSIData(HSID)
start = time.time()
RDHSI = DLMethod(DLM, HSI, NC=k)
end = time.time()
DL_Time = end - start
CRDHSI, CGT = ImageCubes(RDHSI, GT, WS=WS)
Tr, Va, Te, TrC, VaC, TeC = TrTeSplit(CRDHSI, CGT, trRatio, vrRatio, teRatio)
TrC = to_categorical(TrC, num_classes=Num_Classes)
VaC = to_categorical(VaC, num_classes=Num_Classes)
TeC = to_categorical(TeC, num_classes=Num_Classes)
print(Te.shape)
print(TeC.shape)

try:
    train_dataset = tf.data.Dataset.from_tensor_slices((Tr, TrC)).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((Va, VaC)).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((Te, TeC)).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)
    dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    dist_val_dataset = strategy.experimental_distribute_dataset(val_dataset)
    dist_test_dataset = strategy.experimental_distribute_dataset(test_dataset)
except Exception as e:
    print(f"Error creating datasets: {e}")
    sys.exit(1)

# Custom Layers (unchanged from original code)
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

class SplineLinear(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, init_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.init_scale = init_scale
        self.weight = self.add_weight(
            shape=(self.out_features, self.in_features),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            regularizer=tf.keras.regularizers.l2(0.001)
        )
    def call(self, inputs):
        return tf.matmul(inputs, tf.transpose(self.weight))

class RadialBasisFunction(tf.keras.layers.Layer):
    def __init__(self, grid_min=-1.0, grid_max=1.0, num_grids=4, denominator=None, **kwargs):
        super().__init__(**kwargs)
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
        self.grid = tf.Variable(
            initial_value=tf.linspace(grid_min, grid_max, num_grids),
            trainable=False,
            name="grid"
        )
    def call(self, inputs):
        expanded_inputs = tf.expand_dims(inputs, axis=-1)
        return tf.exp(-((expanded_inputs - self.grid) / self.denominator) ** 2)

class FastKANLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, grid_min=-1.0, grid_max=1.0, num_grids=4, use_base_update=True, use_layernorm=True, base_activation=tf.nn.swish, spline_weight_init_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_layernorm = use_layernorm
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)
        else:
            self.layernorm = None
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, init_scale=spline_weight_init_scale)
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
            spline_basis,
            shape=(-1, spline_basis.shape[-2] * spline_basis.shape[-1])
        )
        ret = self.spline_linear(spline_basis_flat)
        output_shape = tf.concat([tf.shape(inputs)[:-1], [self.output_dim]], axis=0)
        ret = tf.reshape(ret, shape=output_shape)
        if self.use_base_update:
            base = self.base_linear(self.base_activation(inputs))
            ret = ret + base
        return ret

class MultiHeadKANAttention(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_heads, output_dim, grid_min=-1.0, grid_max=1.0, num_grids=4, use_base_update=True, use_layernorm=True, base_activation=tf.nn.swish, spline_weight_init_scale=0.1, dropout_rate=0.5, output_activation=None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.output_dim = output_dim
        self.all_head_size = self.num_heads * self.head_dim
        self.query_kan = FastKANLayer(input_dim=input_dim, output_dim=self.all_head_size, grid_min=grid_min, grid_max=grid_max, num_grids=num_grids, use_base_update=use_base_update, use_layernorm=use_layernorm, base_activation=base_activation, spline_weight_init_scale=spline_weight_init_scale)
        self.key_kan = FastKANLayer(input_dim=input_dim, output_dim=self.all_head_size, grid_min=grid_min, grid_max=grid_max, num_grids=num_grids, use_base_update=use_base_update, use_layernorm=use_layernorm, base_activation=base_activation, spline_weight_init_scale=spline_weight_init_scale)
        self.value_kan = FastKANLayer(input_dim=input_dim, output_dim=self.all_head_size, grid_min=grid_min, grid_max=grid_max, num_grids=num_grids, use_base_update=use_base_update, use_layernorm=use_layernorm, base_activation=base_activation, spline_weight_init_scale=spline_weight_init_scale)
        self.output_projection = tf.keras.layers.Dense(units=32, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.output_activation = tf.keras.layers.Activation(output_activation) if output_activation else None
        self.attention_dropout = tf.keras.layers.Dropout(dropout_rate)
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def combine_heads(self, x):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (tf.shape(x)[0], -1, self.num_heads * self.head_dim))
    def call(self, queries, keys, values, training=False):
        batch_size = tf.shape(queries)[0]
        Q = self.query_kan(queries)
        K = self.key_kan(keys)
        V = self.value_kan(values)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        attention_scores = tf.matmul(Q, K, transpose_b=True)
        d_k = tf.cast(self.head_dim, tf.float32)
        attention_scores /= tf.math.sqrt(d_k)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.attention_dropout(attention_weights, training=training)
        attention_output = tf.matmul(attention_weights, V)
        attention_output = self.combine_heads(attention_output)
        output = self.output_projection(attention_output)
        if self.output_activation:
            output = self.output_activation(output)
        return output

class SpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
        self.spatial_gate = Sequential([
            FastKANLayer(input_dim=out_channels, output_dim=out_channels, grid_min=-1.0, grid_max=1.0, num_grids=4),
            Activation('sigmoid'),
        ])
        self.spectral_gate = Sequential([
            FastKANLayer(input_dim=out_channels, output_dim=out_channels, grid_min=-1.0, grid_max=1.0, num_grids=4),
            Activation('sigmoid'),
        ])
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
    def __init__(self, out_channels, state_dim=64, num_heads=num_heads, num_grids=4, **kwargs):
        super(SSFKSMambaModel, self).__init__(**kwargs)
        self.token_generation = SpectralSpatialTokenGeneration(out_channels)
        self.multi_head_attention = MultiHeadKANAttention(input_dim=out_channels, output_dim=out_channels, num_heads=num_heads, grid_min=-1.0, grid_max=1.0, num_grids=num_grids, base_activation=tf.nn.swish)
        self.feature_enhancement = SpectralSpatialFeatureEnhancement(out_channels)
        self.state_space_model = StateSpaceModel(state_dim=state_dim)
        self.dense = tf.keras.layers.Dense(units=out_channels, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.classifier = tf.keras.layers.Dense(Num_Classes, activation='softmax')
    def call(self, x, training=False):
        spatial_tokens, spectral_tokens = self.token_generation(x)
        center_tokens = spatial_tokens[:, x.shape[1] // 2, :]
        spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
        spatial_attention = self.multi_head_attention(spatial_enhanced, spatial_enhanced, spatial_enhanced)
        spectral_attention = self.multi_head_attention(spectral_enhanced, spectral_enhanced, spectral_enhanced)
        combined_output = tf.concat([spatial_attention, spectral_attention], axis=-1)
        state_output = self.state_space_model(combined_output)
        output = self.classifier(state_output)
        return output
    def get_embeddings(self, x):
        spatial_tokens, spectral_tokens = self.token_generation(x)
        center_tokens = spatial_tokens[:, x.shape[1] // 2, :]
        spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
        spatial_attention = self.multi_head_attention(spatial_enhanced, spatial_enhanced, spatial_enhanced)
        spectral_attention = self.multi_head_attention(spectral_enhanced, spectral_enhanced, spectral_enhanced)
        combined_output = tf.concat([spatial_attention, spectral_attention], axis=-1)
        state_output = self.state_space_model(combined_output)
        dense_output = self.dense(state_output)
        return dense_output

def FastKanMamba(Tr, batch_size):
    model = SSFKSMambaModel(out_channels=out_channels, state_dim=64, num_heads=num_heads, num_grids=4)
    _ = model(Tr[:batch_size])
    return model

def get_flops(model):
    total_flops = 0
    def process_layer(layer):
        nonlocal total_flops
        flops = 0
        if isinstance(layer, tf.keras.layers.Conv2D):
            input_channels = layer.input_shape[-1] if layer.input_shape is not None else 0
            output_height = layer.output_shape[1] if layer.output_shape is not None else 0
            output_width = layer.output_shape[2] if layer.output_shape is not None else 0
            kernel_size = layer.kernel_size[0] * layer.kernel_size[1] * input_channels
            flops += kernel_size * output_height * output_width * layer.filters
        elif isinstance(layer, tf.keras.layers.Dense):
            output_dim = layer.units
            input_dim = layer.get_weights()[0].shape[0] if layer.get_weights() else 0
            flops += 2 * input_dim * output_dim
            if layer.activation is not None and layer.activation.__name__ != 'linear':
                flops += input_dim
        elif isinstance(layer, SplineLinear):
            flops += layer.in_features * layer.out_features
        elif isinstance(layer, RadialBasisFunction):
            input_dim = layer.get_weights()[0].shape[0] if layer.get_weights() else 0
            flops += 4 * input_dim * layer.num_grids
        elif isinstance(layer, FastKANLayer):
            input_dim = layer.input_dim
            output_dim = layer.output_dim
            num_grids = layer.rbf.num_grids
            if layer.use_layernorm:
                flops += 2 * input_dim
            rbf_flops = 4 * input_dim * num_grids
            spline_flops = input_dim * num_grids * output_dim
            base_flops = 0
            if layer.use_base_update:
                base_flops += input_dim * output_dim
                base_flops += input_dim
            flops += rbf_flops + spline_flops + base_flops
        elif isinstance(layer, SpectralSpatialTokenGeneration):
            flops += process_layer(layer.spatial_tokens)
            flops += process_layer(layer.spectral_tokens)
        elif isinstance(layer, MultiHeadKANAttention):
            flops += process_layer(layer.query_kan)
            flops += process_layer(layer.key_kan)
            flops += process_layer(layer.value_kan)
            flops += process_layer(layer.output_projection)
            if layer.output_activation:
                flops += layer.output_dim
        elif isinstance(layer, SpectralSpatialFeatureEnhancement):
            flops += process_layer(layer.spatial_gate)
            flops += process_layer(layer.spectral_gate)
        elif isinstance(layer, StateSpaceModel):
            flops += process_layer(layer.state_transition)
            flops += process_layer(layer.state_update)
        elif isinstance(layer, tf.keras.Sequential) or hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                flops += process_layer(sublayer)
        elif isinstance(layer, tf.keras.Model):
            for sublayer in layer.layers:
                flops += process_layer(sublayer)
        total_flops += flops
        print(f"Layer {layer.name}: {flops} FLOPs")
        return flops
    process_layer(model)
    return total_flops

def compute_flops(model):
    return get_flops(model)

def train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, CRDHSI, HSID, teRatio, k, WS, DLM, RDHSI, GT, batch_size, epochs, output_dir):
    print(f'Model Name: {model_name}')
    output_dir = os.path.join(output_dir, "") + model_name.__name__
    os.makedirs(output_dir, exist_ok=True)
    print(model_name.__name__)

    try:
        if model_name == FastKanMamba:
            Tr = Tr.reshape(-1, WS, WS, k)
            Va = Va.reshape(-1, WS, WS, k)
            Te = Te.reshape(-1, WS, WS, k)

        with strategy.scope():
            model = FastKanMamba(Tr, batch_size)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        file_name = f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_model_summary_of_{model_name.__name__}.txt"
        total_params = model.count_params()
        print("Calculating FLOPs...")
        model_flops = compute_flops(model)
        with open(os.path.join(output_dir, file_name), 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write(f"\nTotal Parameters: {total_params}\n")
            fh.write(f"Total FLOPs: {model_flops}\n")

        start = time.time()
        history = model.fit(
            dist_train_dataset,
            epochs=epochs,
            validation_data=dist_val_dataset,
            verbose=1
        )
        end = time.time()
        Tr_Time = end - start

        start = time.time()
        Te_Pre = model.predict(dist_test_dataset)
        end = time.time()
        Te_Time = end - start

        classification, Confusion, OA, Per_Class, AA, Kappa = ClassificationReports(TeC, Te_Pre, target_names, zero_division=0)
        Te_Kappa = round(Kappa, 4)
        file_name = os.path.join(output_dir, f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_{batch_size}_Classification_Report.csv")
        CSVResults(file_name, classification, Confusion, Tr_Time, Te_Time, DL_Time, Kappa, OA, AA, Per_Class)
        print(classification)

        embeddings = model.get_embeddings(Tr)
        embeddings_np = embeddings.numpy()
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_np)
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 5))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=TrC.argmax(axis=1), cmap='viridis')
        plt.title(f'tSNE: {HSID}', fontsize=12, fontweight='bold')
        plt.xlabel('tSNE-1', fontsize=12, fontweight='bold')
        plt.ylabel('tSNE-2', fontsize=12, fontweight='bold')
        plt.tick_params(labelsize=12)
        plt.grid(True)
        plt.tight_layout()
        file_name = f"{HSID}_{trRatio}_{k}_{WS}_tsne.png"
        plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)

        outputs = GT_Plot(CRDHSI, GT, model, WS, k)
        plt.figure(figsize=(10, 10))
        plt.imshow(outputs, cmap='nipy_spectral')
        plt.axis('off')
        file_name = f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_{batch_size}_Ground_Truths.png"
        plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)

        return history, Tr_Time, Te_Time, OA, AA, Kappa, total_params
    except Exception as e:
        print(f"Error in train_and_evaluate_model: {e}")
        sys.exit(1)
    finally:
        tf.keras.backend.clear_session()
# Main execution
# Main execution
try:
    model_names = [FastKanMamba]
    all_results = {}
    history_list = []
    Tr_Time_list = []
    Te_Time_list = []

    for model_name in model_names:
        history, Tr_Time, Te_Time, OA, AA, Kappa, total_params = train_and_evaluate_model(
            model_name, Tr, TrC, Va, VaC, Te, TeC, adam, CRDHSI, HSID, teRatio, k, WS, DLM, RDHSI, GT, batch_size, epochs, output_dir
        )
        history_list.append(history)
        Tr_Time_list.append(Tr_Time)
        Te_Time_list.append(Te_Time)
        model_function_name = model_name.__name__
        all_results[model_function_name] = {
            "history": history.history,
            "Tr_Time": Tr_Time,
            "Te_Time": Te_Time,
            "Overall Accuracy": OA,
            "Average Accuracy": AA,
            "Kappa": Kappa,
            "Training Parameters": total_params
        }

    all_results_filename = "all_models_results.json"
    all_results_output_path = os.path.join(output_dir, all_results_filename)
    json_data = json.dumps(all_results, indent=4)
    with open(all_results_output_path, 'w') as json_file:
        json_file.write(json_data)

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    colors = ['brown', 'red', 'blue', 'gray', 'green', 'purple', 'orange']
    for i, history in enumerate(history_list):
        axs[0].plot(history.history['loss'], label=f'{model_names[i].__name__} Train', color=colors[i])
        axs[0].plot(history.history['val_loss'], label=f'{model_names[i].__name__} Val', color=colors[i], linestyle='--')
        axs[1].plot(history.history['accuracy'], label=f'{model_names[i].__name__} Train', color=colors[i])
        axs[1].plot(history.history['val_accuracy'], label=f'{model_names[i].__name__} Val', color=colors[i], linestyle='--')
    axs[0].set_title('Loss', fontsize=10)
    axs[0].set_xlabel('Epoch', fontsize=10)
    axs[1].set_title('Accuracy', fontsize=10)
    axs[1].set_xlabel('Epoch', fontsize=10)
    axs[1].legend(fontsize=6)
    axs[0].grid(True)
    axs[1].grid(True)
    plt.tight_layout()
    file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_acc_loss_curve_all_models.png"
    plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)
except Exception as e:
    print(f"Error in main execution: {e}")
    sys.exit(1)
finally:
    tf.keras.backend.clear_session()