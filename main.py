# Libraries
import time
import json
import os, argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import numpy as np
import random as _random

# Matplotlib (visualization)
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Sklearrn,Scipy, seaborn
import scipy.io as sio  # MATLAB data file
import seaborn as sns
from scipy.signal import medfilt  # Scipy signal process library (median filter)
from sklearn.manifold import TSNE

# Spectrak to Load HSI Dataset
import spectral
import spectral.io.envi as envi
from typing import List
from operator import truediv

# Keras
import keras
from keras.losses import categorical_crossentropy
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.optimizers import legacy
from models.vit import SSFormer, PyFormer, WaveFormer

# TensorFlow
import tensorflow as tf

tf.keras.backend.clear_session()
from tensorflow.keras import layers, Model
from keras.utils import to_categorical
from keras.layers import Dense
from tensorflow.keras.layers import Activation, Lambda, multiply
from tensorflow.keras.optimizers import Adam
from utils import *

parser = argparse.ArgumentParser(description="Script Description")
# Add arguments for each parameter
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    choices=["LK", "HC", "UH13", "PU", "PC", "SA", "HH", "QUH", "PI", "TH", "Fe"],
    default="PI",
    help="Dataset identifier",
)
parser.add_argument("--window_size", "-ws", type=int, default=4, help="Window size")
parser.add_argument(
    "--dl_method", "-dlm", type=str, default="PCA", help="Dimension Reduction Method"
)  # 차원 축소 메소드
parser.add_argument("--test_ratio", "-te", type=float, default=0.90, help="Test ratio")
parser.add_argument(
    "--train_ratio", "-tr", type=float, default=0.50, help="Train ratio"
)
parser.add_argument(
    "--validation_ratio", "-vr", type=float, default=0.50, help="Validation ratio"
)
parser.add_argument(
    "--number_of_bands",
    "-k",
    type=int,
    default=15,
    help="Number of bands principal components for dimensionality reduction",
)
parser.add_argument(
    "--seed", type=int, default=0, help="Seed for split, init and shuffling"
)
parser.add_argument(
    "--epochs", "-e", type=int, default=50, help="Number of epochs for training"
)
parser.add_argument(
    "--batch_size", "-bs", type=int, default=56, help="Batch size for training"
)
# Parse the command-line arguments
args = parser.parse_args()


def set_seed(seed):
    _random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)  # python + numpy + tf


set_seed(args.seed)
# DataSet Loader Funtion
base_dir = "./datasets"


def LoadHSIData(method):
    """
    데이터셋 이름을 받아서 데이터 로드하는 함수수
    method: str : 'LK', 'HC', 'UH13', 'PU', 'PC', 'SA', 'BS', 'HH', 'QUH', 'PI', 'TH'
    dataset name
    return: HSI, GT, Num_Classes, target_names
    데이터, 정답, 클래스 수, 클래스 이름
    """

    data_path = os.path.join(os.getcwd(), f"{base_dir}")
    if method == "LK":
        ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
        HSI = sio.loadmat(os.path.join(data_path, "WHU_Hi_LongKou.mat"))[
            "WHU_Hi_LongKou"
        ]
        GT = sio.loadmat(os.path.join(data_path, "WHU_Hi_LongKou_gt"))[
            "WHU_Hi_LongKou_gt"
        ]
        Num_Classes = 9
        target_names = [
            "Corn",
            "Cotton",
            "Sesame",
            "Broad-leaf soybean",  # class name list
            "Narrow-leaf soybean",
            "Rice",
            "Water",
            "Roads and houses",
            "Mixed weed",
        ]

    elif method == "HH":
        ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
        HSI = sio.loadmat(os.path.join(data_path, "WHU_Hi_HongHu.mat"))["WHU_Hi_HongHu"]
        GT = sio.loadmat(os.path.join(data_path, "WHU_Hi_HongHu_gt"))[
            "WHU_Hi_HongHu_gt"
        ]
        Num_Classes = 22
        target_names = [
            "Red roof",
            "Road",
            "Bare soil",
            "Cotton",
            "Cotton firewood",
            "Rape",
            "Chinese cabbage",
            "Pakchoi",
            "Cabbage",
            "Tuber mustard",
            "Brassica parachinensis",
            "Brassica chinensis",
            "Small Brassica chinensis",
            "Lactuca sativa",
            "Celtuce",
            "Film covered lettuce",
            "Romaine lettuce",
            "Carrot",
            "White radish",
            "Garlic sprout",
            "Broad bean",
            "Tree",
        ]

    elif method == "HC":
        ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
        HSI = sio.loadmat(os.path.join(data_path, "WHU_Hi_HanChuan.mat"))[
            "WHU_Hi_HanChuan"
        ]
        GT = sio.loadmat(os.path.join(data_path, "WHU_Hi_HanChuan_gt"))[
            "WHU_Hi_HanChuan_gt"
        ]
        Num_Classes = 16
        target_names = [
            "Strawberry",
            "Cowpea",
            "Soybean",
            "Sorghum",
            "Water spinach",
            "Watermelon",
            "Greens",
            "Trees",
            "Grass",
            "Red roof",
            "Gray roof",
            "Plastic",
            "Bare soil",
            "Road",
            "Bright object",
            "Water",
        ]

    elif method == "BS":
        HSI = sio.loadmat(os.path.join(data_path, "Botswana.mat"))["Botswana"]
        GT = sio.loadmat(os.path.join(data_path, "Botswana_gt.mat"))["Botswana_gt"]
        Num_Classes = 14
        target_names = [
            "Water",
            "Hippo Grass",
            "Floodplain Grasses 1",
            "Floodplain Grasses 2",
            "Reeds 1",
            "Riparian",
            "Firescar 2",
            "Island Interior",
            "Woodlands",
            "Acacia Shrublands",
            "Acacia Grasslands",
            "Short Mopane",
            "Mixed Mopane",
            "Exposed Soils",
        ]

    elif method == "PU":
        HSI = sio.loadmat(os.path.join(data_path, "PaviaU.mat"))["paviaU"]
        GT = sio.loadmat(os.path.join(data_path, "PaviaU_gt.mat"))["paviaU_gt"]
        Num_Classes = 9
        target_names = [
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted",
            "Soil",
            "Bitumen",
            "Bricks",
            "Shadows",
        ]

    elif method == "PC":
        HSI = sio.loadmat(os.path.join(data_path, "Pavia.mat"))["pavia"]
        GT = sio.loadmat(os.path.join(data_path, "Pavia_gt.mat"))["pavia_gt"]
        Num_Classes = 9
        target_names = [
            "Water",
            "Trees",
            "Asphalt",
            "Bricks",
            "Bitumen",
            "Tiles",
            "Shadows",
            "Meadows",
            "Soil",
        ]

    elif method == "SA":
        HSI = sio.loadmat(os.path.join(data_path, "Salinas_corrected.mat"))[
            "salinas_corrected"
        ]
        GT = sio.loadmat(os.path.join(data_path, "Salinas_gt.mat"))["salinas_gt"]
        Num_Classes = 16
        target_names = [
            "Weeds_1",
            "Weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_Weeds",
            "Lettuce_4wk",
            "Lettuce_5wk",
            "Lettuce_6wk",
            "Lettuce_7wk",
            "Vinyard_untrained",
            "Vinyard_trellis",
        ]

    elif method == "UH13":
        HSI = sio.loadmat(os.path.join(data_path, "HU13.mat"))["HSI"]
        GT = sio.loadmat(os.path.join(data_path, "HU13_gt.mat"))["gt"]
        Num_Classes = 15
        target_names = [
            "Healthy grass",
            "Stressed grass",
            "Synthetic grass",
            "Trees",
            "Soil",
            "Water",
            "Residential",
            "Commercial",
            "Road",
            "Highway",
            "Railway",
            "Parking Lot 1",
            "Parking Lot 2",
            "Tennis Court",
            "Running Track",
        ]

    elif method == "QUH":
        HSI = sio.loadmat(os.path.join(data_path, "QUH-Qingyun.mat"))["Chengqu"]
        GT = sio.loadmat(os.path.join(data_path, "QUH-Qingyun_GT.mat"))["ChengquGT"]
        Num_Classes = 6
        target_names = [
            "Trees",
            "Concrete building",
            "Car",
            "Ironhide building",
            "Plastic playground",
            "Asphalt road",
        ]

    elif method == "PI":
        HSI = sio.loadmat(os.path.join(data_path, "QUH-Pingan.mat"))["Haigang"]
        GT = sio.loadmat(os.path.join(data_path, "QUH-Pingan_GT.mat"))["HaigangGT"]
        Num_Classes = 10
        target_names = [
            "Ship",
            "Seawater",
            "Trees",
            " Concrete structure building",
            "Floating pier",
            "Brick houses",
            "Steel houses",
            " Wharf construction land",
            "Car",
            "Road",
        ]

    elif method == "TH":
        HSI = sio.loadmat(os.path.join(data_path, "QUH-Tangdaowan.mat"))["Tangdaowan"]
        GT = sio.loadmat(os.path.join(data_path, "QUH-Tangdaowan_GT.mat"))[
            "TangdaowanGT"
        ]
        Num_Classes = 18
        target_names = [
            "Rubber track",
            "Flaggingv",
            "Sandy",
            "Asphalt",
            "Boardwalk",
            "Rocky shallows",
            "Grassland",
            "Bulrush",
            "Gravel road",
            "Ligustrum vicaryi",
            "Coniferous pine",
            "Spiraea",
            "Bare soil",
            "Buxus sinica",
            "Photinia serrulata",
            "Populus",
            "Ulmus pumila L",
            "Seawater",
        ]

    elif method == "Fe":
        HSI = sio.loadmat(os.path.join(data_path, "fei2.mat"))["Y_whole_band"]
        GT = sio.loadmat(os.path.join(data_path, "FeiC_gt.mat"))["patch_gt"]
        Num_Classes = 20
        target_names = [
            "Crops 1",
            "Bare Soil",
            "Shrubs",
            "Playground track",
            "Road",
            "Trees",
            "Crops 2",
            "Shadow",
            "Concrete floor",
            "Concrete roof",
            "Caigang watt roof",
            "Cars",
            "Asphalt roof ",
            "Grass",
            "Glazed tile roof",
            "Clay tile roof ",
            "Special material",
            "Piazza area ",
            "White tile floor",
            "Black tile floor",
        ]
    else:
        print("Wrong Choice")

    return HSI, GT, Num_Classes, target_names


# Global Parameters
HSID = (
    args.dataset
)  ## "SLA", "IP", "PU", "PC", "SA", "KSC", "BS", "LK", "HH" (difficult to compile), "HC"
DLM = args.dl_method  ## "PCA", "iPCA"
WS = args.window_size
# Using the calculated values
trRatio = args.train_ratio
vrRatio = args.validation_ratio
teRatio = args.test_ratio
k = args.number_of_bands
num_heads = 8
out_channels = 32
epochs = args.epochs
batch_size = args.batch_size
lr = 0.0001
adam = tf.keras.optimizers.legacy.Adam(learning_rate=lr, decay=1e-06, clipnorm=1.0)
# output_dir = os.path.join(f"Spaital_Spectral_Kan_Mamba/{HSID}/window_size/{WS}")
output_dir = os.path.join(
    f"FKANMamba_{num_heads}_chnal_{out_channels}_bs_{batch_size}_{trRatio}/{HSID}/{WS}"
)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# %%
## Creat Patches for 3D (Spatial-Spectral) Models
def ImageCubes(HSI, GT, WS=WS, removeZeroLabels=True):
    """
    HSI 데이터를 Image Cubes로 변환하는 함수. Image Cubes는 3D 데이터로, 각 픽셀 주변의 윈도우 크기만큼의 데이터를 포함

    HSI: np.array : hyperspectral image
    GT: np.array : ground truth
    WS: int : window size
    removeZeroLabels: bool : remove zero labels
    """
    num_rows, num_cols, num_bands = HSI.shape  # HSI data shape (width, height, bands)
    margin = int(WS / 2)  # padding margin
    padded_data = np.pad( HSI, ((margin, margin), (margin, margin), (0, 0)), mode="constant")  # zero padding
    image_cubes = np.zeros( (num_rows * num_cols, WS, WS, num_bands))  # array initialize with zeros for image cubes
    patchesLabels = np.zeros((num_rows * num_cols))  # array initialize with zeros for patches labels (number of pixels)
    patchIndex = 0  # patch index initialize with zero
    for r in range(margin, num_rows + margin):
        for c in range(margin, num_cols + margin):
            cube = padded_data[r - margin : r + margin, c - margin : c + margin, :]  #  Extract a window size cube centred on the current pixel
            image_cubes[patchIndex, :, :, :] = (cube)  # Assign the cube to the image cubes array
            patchesLabels[patchIndex] = GT[r - margin, c - margin]  # Assign the ground truth label to the patches labels array
            patchIndex = patchIndex + 1  # Increment the patch index
    if removeZeroLabels:  # Remove the zero labels
        image_cubes = image_cubes[ patchesLabels > 0, :, :, :]  # Remove the zero labels from the image cubes
        patchesLabels = patchesLabels[patchesLabels > 0]  # Remove the zero labels from the patches labels
        patchesLabels -= 1  # Subtract 1 from the patches labels to shift the class labels
    return image_cubes, patchesLabels  # Return the image cubes and patches labels


# %%
HSI, GT, Num_Classes, target_names = LoadHSIData(HSID)  # Load the HSI data
## Reduce the Dimensionality
# if k < HSI.shape[2]:
start = time.time()
RDHSI = DLMethod(DLM, HSI, NC=k)  # Reduce the dimensionality of the HSI data
end = time.time()
DL_Time = end - start
## Create Image Cubes for Model Building
CRDHSI, CGT = ImageCubes(RDHSI, GT, WS=WS)
## Split Train/validation and Test sets
Tr, Va, Te, TrC, VaC, TeC = TrTeSplit(CRDHSI, CGT, trRatio, vrRatio, teRatio)

# total_samples = Tr.shape[0] + Va.shape[0] + Te.shape[0]
# print(f"{HSID}Total number of samples:{total_samples}")
# exit(0)
# Reshape train, validation, and test sets
# Tr = Tr.reshape(-1, WS, WS, k, 1)
TrC = to_categorical(
    TrC, num_classes=Num_Classes
)  # Convert the train labels to categorical (one-hot encoding)
# Va = Va.reshape(-1, WS, WS, k, 1)
VaC = to_categorical(VaC, num_classes=Num_Classes)
# Te = Te.reshape(-1, WS, WS, k, 1)
TeC = to_categorical(TeC, num_classes=Num_Classes)
print(Te.shape)
print(TeC.shape)


# Token Generation
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


# Fast Kolomgorov Arnold Network
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


# Radial Basis Function for Fast KAN
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


# Fast Kolmogorov Arnold Network Layer
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


# 멀티 헤드 어텐션을 Fast KAN으로 구현
# SpatialSpectralAttention With Fast Kolmogorov Arnold Network
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


# State Space Model
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


# 위에 나온 것들 모두 사용한 전체 모델
# Architecture
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


# 모델 생성 함수
def FastKanMamba(Tr, batch_size):
    model = SSFKSMambaModel(
        out_channels=out_channels,
        state_dim=64,
        num_heads=num_heads,
        num_grids=8,
    )
    # Build the model by passing a batch of data
    _ = model(Tr[:batch_size])  # Ensures model is built correctly
    return model


# Flops Calculation
def get_flops(model):
    total_flops = 0

    def process_layer(layer):
        nonlocal total_flops
        flops = 0

        # Handle Conv2D layers
        if isinstance(layer, tf.keras.layers.Conv2D):
            input_channels = (
                layer.input_shape[-1] if layer.input_shape is not None else 0
            )
            output_height = (
                layer.output_shape[1] if layer.output_shape is not None else 0
            )
            output_width = (
                layer.output_shape[2] if layer.output_shape is not None else 0
            )
            kernel_size = layer.kernel_size[0] * layer.kernel_size[1] * input_channels
            flops += kernel_size * output_height * output_width * layer.filters

        # Handle Dense layers
        elif isinstance(layer, tf.keras.layers.Dense):
            output_dim = layer.units
            # Use weights to infer input_dim if input_shape is unavailable
            input_dim = layer.get_weights()[0].shape[0] if layer.get_weights() else 0
            flops += 2 * input_dim * output_dim  # Multiplications + additions
            if layer.activation is not None and layer.activation.__name__ != "linear":
                flops += input_dim  # Approximate activation cost

        # Handle SplineLinear
        elif isinstance(layer, SplineLinear):
            flops += layer.in_features * layer.out_features

        # Handle RadialBasisFunction
        elif isinstance(layer, RadialBasisFunction):
            input_dim = layer.get_weights()[0].shape[0] if layer.get_weights() else 0
            flops += (
                4 * input_dim * layer.num_grids
            )  # Subtraction, squaring, division, exp

        # Handle FastKANLayer
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

        # Handle SpectralSpatialTokenGeneration
        elif isinstance(layer, SpectralSpatialTokenGeneration):
            flops += process_layer(layer.spatial_tokens)  # Dense layer
            flops += process_layer(layer.spectral_tokens)  # Dense layer

        # Handle FastKANSpatialSpectralAttention
        elif isinstance(layer, FastKANSpatialSpectralAttention):
            flops += process_layer(layer.query_kan)  # FastKANLayer
            flops += process_layer(layer.key_kan)  # FastKANLayer
            flops += process_layer(layer.value_kan)  # FastKANLayer
            flops += process_layer(layer.output_projection)  # Dense layer
            if layer.output_activation:
                flops += layer.output_dim  # Activation cost
        # Handle SpectralSpatialFeatureEnhancement
        elif isinstance(layer, SpectralSpatialFeatureEnhancement):
            flops += process_layer(
                layer.spatial_gate
            )  # Sequential with FastKANLayer and Activation
            flops += process_layer(
                layer.spectral_gate
            )  # Sequential with FastKANLayer and Activation

        # Handle StateSpaceModel
        elif isinstance(layer, StateSpaceModel):
            flops += process_layer(layer.state_transition)  # Dense layer
            flops += process_layer(layer.state_update)  # Dense layer

        # Handle Sequential or nested models
        elif isinstance(layer, tf.keras.Sequential) or hasattr(layer, "layers"):
            for sublayer in layer.layers:
                flops += process_layer(sublayer)

        # Recursively process tf.keras.Model
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


# Train and Evaluate Model
def train_and_evaluate_model(
    model_name,
    Tr,
    TrC,
    Va,
    VaC,
    Te,
    TeC,
    adam,
    CRDHSI,
    HSID,
    teRatio,
    k,
    WS,
    DLM,
    RDHSI,
    GT,
    batch_size,
    epochs,
    output_dir,
):
    """
    모델 명을 받아서 모델을 불러오고 입력된 데이터와 하이퍼파라미터를 바탕으로 모델을 학습하는 함수
    model_name: keras model : keras model
    Tr: np.array : train data
    TrC: np.array : train labels
    Va: np.array : validation data
    VaC: np.array : validation labels
    Te: np.array : test data
    TeC: np.array : test labels
    adam: keras optimizer : keras optimizer
    CRDHSI: np.array : data?
    HSID: str : data name?
    teRatio: float : test ratio
    k: int : number of bands
    WS: int : window size
    DLM: str : dimensionality reduction method
    RDHSI: np.array : data
    GT: np.array : ground truth
    batch_size: int : batch size
    epochs: int : number of epochs
    output_dir: str : output directory

    return
    history: keras history : keras history
    Tr_Time: float : training time
    Te_Time: float : test time
    OA: float : overall accuracy
    AA: float : average accuracy
    Kappa: float : kappa coefficient
    total_params: int : total number of parameters
    """
    print(f"Model Name : {model_name}")
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(output_dir, "") + model_name.__name__
    os.makedirs(output_dir, exist_ok=True)
    print(model_name.__name__)

    # 모델 명이 FastKanMamba인 경우 dataset을 reshape
    if model_name == FastKanMamba:
        Tr = Tr.reshape(-1, WS, WS, k)  # Flatten input for the model
        Va = Va.reshape(-1, WS, WS, k)  # Flatten input for the model
        Te = Te.reshape(-1, WS, WS, k)
        model = FastKanMamba(Tr, batch_size)
    else:
        model = model_name(WS, k, Num_Classes)

    # Model Summary
    file_name = f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_model_summary_of_{model_name.__name__}.txt"
    # Assuming `model` is your Keras model
    total_params = model.count_params()
    print("Calculating FLOPs...")
    model_flops = compute_flops(model)
    with open(os.path.join(output_dir, file_name), "w") as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + "\n"))
        fh.write(f"\nTotal Parameters: {total_params}\n")
        fh.write(f"Total FLOPs: {model_flops}\n")

    # Compiling the model (모델의 loss, optimizer, metrics 설정)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
    # Training the model
    start = time.time()
    history = model.fit(
        x=Tr, y=TrC, batch_size=batch_size, epochs=epochs, validation_data=(Va, VaC)
    )
    end = time.time()
    Tr_Time = end - start

    # Test Model
    start = time.time()
    Te_Pre = model.predict(Te)
    end = time.time()
    Te_Time = end - start

    ## Classification Report for Test Model
    classification, Confusion, OA, Per_Class, AA, Kappa = ClassificationReports(
        TeC, Te_Pre, target_names, zero_division=0
    )
    Te_Kappa = round(Kappa, 4)
    # Writing Results in CSV File
    file_name = os.path.join(
        output_dir,
        f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_{batch_size}_Classification_Report.csv",
    )
    CSV_file = CSVResults(
        file_name,
        classification,
        Confusion,
        Tr_Time,
        Te_Time,
        DL_Time,
        Kappa,
        OA,
        AA,
        Per_Class,
    )
    print(classification)

    # Ground Truths Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(GT, cmap="nipy_spectral")
    plt.axis("off")
    file_name = f"{HSID}_Ground_Truths.png"
    plt.savefig(
        os.path.join(output_dir, file_name),
        dpi=500,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    outputs = GT_Plot(CRDHSI, GT, model, WS, k)
    plt.figure(figsize=(10, 10))
    plt.imshow(outputs, cmap="nipy_spectral")
    plt.axis("off")
    file_name = (
        f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_{batch_size}_Ground_Truths.png"
    )
    plt.savefig(
        os.path.join(output_dir, file_name),
        dpi=500,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )

    return history, Tr_Time, Te_Time, OA, AA, Kappa, total_params


model_names = [FastKanMamba]
all_results = {}
history_list = []
Tr_Time_list = []
Te_Time_list = []

for model_name in model_names:
    # Train and Evaluate Model
    history, Tr_Time, Te_Time, OA, AA, Kappa, total_params = train_and_evaluate_model(
        model_name,
        Tr,
        TrC,
        Va,
        VaC,
        Te,
        TeC,
        adam,
        CRDHSI,
        HSID,
        teRatio,
        k,
        WS,
        DLM,
        RDHSI,
        GT,
        batch_size,
        epochs,
        output_dir,
    )
    history_list.append(history)
    Tr_Time_list.append(Tr_Time)
    Tr_Time_list.append(Te_Time)

    # Save the results in a dictionary
    model_function_name = model_name.__name__
    all_results[model_function_name] = {
        "history": history.history,
        "Tr_Time": Tr_Time,
        "Te_Time": Te_Time,
        "Overall Accuracy": OA,
        "Average Accuracy": AA,
        "Kappa": Kappa,
        "Training Parameters": total_params,
    }

# Define the filename for all results
all_results_filename = "all_models_results.json"

# Concatenate output_dir with the filename
all_results_output_path = os.path.join(output_dir, all_results_filename)

# Convert the dictionary to JSON format
json_data = json.dumps(all_results, indent=4)

# Optionally, write the JSON data to a file
with open(all_results_output_path, "w") as json_file:
    json_file.write(json_data)

# Plot training and validation loss and accuracy for each model
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
colors = ["brown", "red", "blue", "gray", "green", "purple", "orange"]
for i, history in enumerate(history_list):
    # Plot loss
    axs[0].plot(
        history.history["loss"],
        label=f"{model_names[i].__name__} Train",
        color=colors[i],
    )
    axs[0].plot(
        history.history["val_loss"],
        label=f"{model_names[i].__name__} Val",
        color=colors[i],
        linestyle="--",
    )
    # Plot accuracy
    axs[1].plot(
        history.history["accuracy"],
        label=f"{model_names[i].__name__} Train",
        color=colors[i],
    )
    axs[1].plot(
        history.history["val_accuracy"],
        label=f"{model_names[i].__name__} Val",
        color=colors[i],
        linestyle="--",
    )
axs[0].set_title("Loss", fontsize=10)
axs[0].set_xlabel("Epoch", fontsize=10)
axs[1].set_title("Accuracy", fontsize=10)
axs[1].set_xlabel("Epoch", fontsize=10)
axs[1].legend(fontsize=6)
axs[0].grid(True)
axs[1].grid(True)
plt.tight_layout()
file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_acc_loss_curve_all_models.png"
plt.savefig(
    os.path.join(output_dir, file_name),
    dpi=500,
    format="png",
    bbox_inches="tight",
    pad_inches=0,
)
tf.keras.backend.clear_session()
