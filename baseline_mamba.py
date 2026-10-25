import os,argparse, json
import time
import warnings
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
import spectral
import spectral.io.envi as envi
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import keras
import pywt
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import  Conv1D, MaxPooling1D, Flatten, Reshape
from keras.layers import (Conv3D, Dense,Input, Reshape,LayerNormalization)
from tensorflow.keras.layers import (Input, InputSpec, Add, AveragePooling2D, LeakyReLU, Multiply, 
                                     Conv3D, Conv2D, Conv1D, BatchNormalization,MaxPool2D,Dropout,Activation,
                                     GlobalAveragePooling2D,
                                     Layer, Flatten, Reshape)
from keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
from keras.models import Model, Sequential
from sklearn.manifold import TSNE
# from vit_keras import utils, vit
from keras.utils import to_categorical
from keras_cv_attention_models import attention_layers
import matplotlib.font_manager as font_manager
from utils import *
import warnings
from models.vit import SSFormer, PyFormer, WaveFormer
from models.vit import wavelet_transform_image
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Script Description")
# Add arguments for each parameter
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    choices=["LK", "HC", "UH13", "PU", "PC", "SA","HH","QUH","PI","TH","Fe"],
    default="SA",
    help="Dataset identifier",
)
parser.add_argument("--window_size", "-ws", type=int, default=12, help="Window size")    
parser.add_argument("--dl_method", "-dlm", type=str, default='PCA', help="Dimension Reduction Method")  # 차원 축소 메소드
parser.add_argument("--test_ratio", "-te", type=float, default=0.90, help="Test ratio")
parser.add_argument("--train_ratio", "-tr", type=float, default=0.50, help="Train ratio")
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
    "--epochs", "-e", type=int, default=5, help="Number of epochs for training"
)
parser.add_argument(
    "--batch_size", "-bs", type=int, default=56, help="Batch size for training"
)
# Parse the command-line arguments
args = parser.parse_args()

# DataSet Loader Funtion
base_dir = './datasets'
def LoadHSIData(method):
    '''
    데이터셋 이름을 받아서 데이터 로드하는 함수수
    method: str : 'LK', 'HC', 'UH13', 'PU', 'PC', 'SA', 'BS', 'HH', 'QUH', 'PI', 'TH'
    dataset name
    return: HSI, GT, Num_Classes, target_names
    데이터, 정답, 클래스 수, 클래스 이름
    '''
    
    data_path = os.path.join(os.getcwd(),f'{base_dir}')
    if method == 'LK':
        ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
        HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt'))['WHU_Hi_LongKou_gt']
        Num_Classes = 9
        target_names = ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',       # class name list
                        'Narrow-leaf soybean', 'Rice', 'Water',
                        'Roads and houses', 'Mixed weed']
        
    elif method == 'HH':
        ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
        HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
        GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_gt'))['WHU_Hi_HongHu_gt']
        Num_Classes = 22
        target_names = ['Red roof', 'Road', 'Bare soil', 'Cotton',
                        'Cotton firewood', 'Rape', 'Chinese cabbage',
                        'Pakchoi', 'Cabbage', 'Tuber mustard', 'Brassica parachinensis',
                        'Brassica chinensis', 'Small Brassica chinensis', 'Lactuca sativa',
                        'Celtuce', 'Film covered lettuce', 'Romaine lettuce',
                        'Carrot', 'White radish', 'Garlic sprout', 'Broad bean',
                        'Tree']
        
    elif method == 'HC':
        ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
        HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
        GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_gt'))['WHU_Hi_HanChuan_gt']
        Num_Classes = 16
        target_names = ['Strawberry', 'Cowpea', 'Soybean', 'Sorghum',
                        'Water spinach', 'Watermelon', 'Greens', 'Trees', 'Grass',
                        'Red roof', 'Gray roof', 'Plastic', 'Bare soil', 'Road',
                        'Bright object', 'Water']
        
    elif method == 'BS':
        HSI = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
        GT = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
        Num_Classes = 14
        target_names = ['Water', 'Hippo Grass', 'Floodplain Grasses 1', 'Floodplain Grasses 2',
                        'Reeds 1', 'Riparian', 'Firescar 2', 'Island Interior', 'Woodlands',
                        'Acacia Shrublands', 'Acacia Grasslands', 'Short Mopane', 'Mixed Mopane', 'Exposed Soils']
     
    elif method == 'PU':
        HSI = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        GT = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        Num_Classes = 9
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted','Soil','Bitumen',
                        'Bricks','Shadows']

    elif method == 'PC':
        HSI = sio.loadmat(os.path.join(data_path, 'Pavia.mat'))['pavia']
        GT = sio.loadmat(os.path.join(data_path, 'Pavia_gt.mat'))['pavia_gt']
        Num_Classes = 9
        target_names = ['Water', 'Trees', 'Asphalt', 'Bricks', 'Bitumen', 'Tiles', 'Shadows',
                        'Meadows', 'Soil']

    elif method == 'SA':
        HSI = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        GT = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        Num_Classes = 16
        target_names = ['Weeds_1','Weeds_2','Fallow',
                        'Fallow_rough_plow','Fallow_smooth', 'Stubble','Celery',
                        'Grapes_untrained','Soil_vinyard_develop','Corn_Weeds',
                        'Lettuce_4wk','Lettuce_5wk','Lettuce_6wk',
                        'Lettuce_7wk', 'Vinyard_untrained','Vinyard_trellis']
    
    elif method == 'UH13':
        HSI = sio.loadmat(os.path.join(data_path, 'HU13.mat'))['HSI']
        GT = sio.loadmat(os.path.join(data_path, 'HU13_gt.mat'))['gt']
        Num_Classes = 15
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees',
                    'Soil', 'Water', 'Residential', 'Commercial', 'Road',
                    'Highway', 'Railway', 'Parking Lot 1', 'Parking Lot 2',
                    'Tennis Court', 'Running Track']
        
    elif method == 'QUH':
        HSI = sio.loadmat(os.path.join(data_path,'QUH-Qingyun.mat'))['Chengqu']
        GT = sio.loadmat(os.path.join(data_path,'QUH-Qingyun_GT.mat'))['ChengquGT']
        Num_Classes = 6
        target_names = ["Trees", "Concrete building", "Car", "Ironhide building", "Plastic playground", "Asphalt road"]
        
    elif method == 'PI':
        HSI = sio.loadmat(os.path.join(data_path,'QUH-Pingan.mat'))['Haigang']
        GT = sio.loadmat(os.path.join(data_path,'QUH-Pingan_GT.mat'))['HaigangGT']
        Num_Classes = 10
        target_names = ["Ship", "Seawater", "Trees"," Concrete structure building", "Floating pier", "Brick houses", 
                      "Steel houses"," Wharf construction land", "Car", "Road"]
        
    elif method == 'TH':
        HSI = sio.loadmat(os.path.join(data_path, 'QUH-Tangdaowan.mat'))['Tangdaowan']
        GT = sio.loadmat(os.path.join(data_path, 'QUH-Tangdaowan_GT.mat'))['TangdaowanGT']
        Num_Classes = 18
        target_names = ['Rubber track', 'Flaggingv', 'Sandy', 'Asphalt', 'Boardwalk', 'Rocky shallows',
            'Grassland', 'Bulrush', 'Gravel road', 'Ligustrum vicaryi', 'Coniferous pine',
            'Spiraea', 'Bare soil', 'Buxus sinica', 'Photinia serrulata', 'Populus',
            'Ulmus pumila L', 'Seawater']
        
    elif method == 'Fe':
        HSI = sio.loadmat(os.path.join(data_path, 'fei2.mat'))['Y_whole_band']
        GT = sio.loadmat(os.path.join(data_path, 'FeiC_gt.mat'))['patch_gt']
        Num_Classes =   20
        target_names = ['Crops 1', 'Bare Soil', 'Shrubs', 'Playground track', 'Road', 'Trees',
                        'Crops 2', 'Shadow', 'Concrete floor', 'Concrete roof', 'Caigang watt roof',
                        'Cars', 'Asphalt roof ', 'Grass', 'Glazed tile roof', 'Clay tile roof ',
                        'Special material', 'Piazza area ', 'White tile floor', 'Black tile floor']
    else:
        print("Wrong Choice")

    return HSI, GT, Num_Classes, target_names

## Global Parameters for Loop
HSID = args.dataset ## "SLA", "IP", "PU", "PC", "SA", "KSC", "BS", "LK", "HH" (difficult to compile), "HC"
DLM = args.dl_method ## "PCA", "iPCA"
WS = args.window_size
# Using the calculated values
teRatio = args.test_ratio
trRatio = args.train_ratio
vrRatio = args.validation_ratio
k = args.number_of_bands
adam = tf.keras.optimizers.legacy.Adam(lr = 0.001, decay = 1e-06)
epochs = args.epochs
batch_size = args.batch_size
num_heads=8
out_channels=32
#output_dir = os.path.join(f"baseline_results_out_32_{num_heads}_{batch_size}/{HSID}/{teRatio}/{WS}")
output_dir = os.path.join(f"baseline_{out_channels}_{num_heads}_{batch_size}/{HSID}/{teRatio}/{WS}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
## Creat Patches for 3D (Spatial-Spectral) Models
def ImageCubes(HSI, GT, WS=WS, removeZeroLabels=True):
    num_rows, num_cols, num_bands = HSI.shape
    margin = int(WS / 2)
    padded_data = np.pad(HSI, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    image_cubes = np.zeros((num_rows * num_cols, WS, WS, num_bands))
    patchesLabels = np.zeros((num_rows * num_cols))
    patchIndex = 0
    for r in range(margin, num_rows + margin):
        for c in range(margin, num_cols + margin):
            cube = padded_data[r - margin: r + margin, c - margin: c + margin, :]
            image_cubes[patchIndex, :, :, :] = cube
            patchesLabels[patchIndex] = GT[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
      image_cubes = image_cubes[patchesLabels>0,:,:,:]
      patchesLabels = patchesLabels[patchesLabels>0]
      patchesLabels -= 1
    return image_cubes, patchesLabels
# **Loading HSI and Creat Patchs**
# Main Function to load Datasets, Dimensional Reduction and Creating Patchs for CNN
HSI, GT, Num_Classes, target_names = LoadHSIData(HSID)
## Reduce the Dimensionality
# if k < HSI.shape[2]:
start = time.time()
RDHSI = DLMethod(DLM, HSI, NC = k)
end = time.time()
DL_Time = end - start
## Create Image Cubes for Model Building
CRDHSI, CGT = ImageCubes(RDHSI, GT, WS = WS)
## Split Train/validation and Test sets
Tr, Va, Te, TrC, VaC, TeC = TrTeSplit(CRDHSI, CGT, trRatio, vrRatio, teRatio)
# Reshape train, validation, and test sets
# Tr = Tr.reshape(-1, WS, WS, k, 1)
TrC = to_categorical(TrC)
# Va = Va.reshape(-1, WS, WS, k, 1)
VaC = to_categorical(VaC)
# Te = Te.reshape(-1, WS, WS, k, 1)
TeC = to_categorical(TeC)

# Attention GCN Starts
def AttentionGCN(WS, k, Num_Classes):

    input_layer = Input(shape=(WS,WS,k), name='input')
    #inputs = data_augmentation(input_layer)
    conv1 = Conv2D(8, kernel_size=1, activation='relu', name='conv1', kernel_regularizer=regularizers.l2(0.0005))(input_layer)
    nn = attention_layers.cot_attention(conv1, kernel_size=3)
    #pool1 = MaxPooling2D(pool_size=(2,2), name='pool1')(nn)
########## Graph-based global reasoning module # input is 8x8x8
    graph_t = Reshape((8, WS*WS))(conv1)
    squeezed_graph_t = Conv1D(16, 1, activation='relu', name='squeezer', kernel_regularizer=regularizers.l2(0.0001))(graph_t)
    squeezed_graph = Reshape((8, 16))(squeezed_graph_t)
    gconv = Conv1D(64, 1, activation='relu', name='gconv', kernel_regularizer=regularizers.l2(0.0001))(squeezed_graph)
    gconv_t = Reshape((8, 64))(gconv)
    unsqueezed_graph_transpose = Conv1D(WS*WS, 1, activation='relu', name='unsqueezer', kernel_regularizer=regularizers.l2(0.0001))(gconv_t)
    unsqueezed_graph = Reshape((WS*WS, 8))(unsqueezed_graph_transpose)
    glore = Reshape((WS, WS, 8))(unsqueezed_graph)

    block1 = layers.concatenate([conv1, glore], axis=-1) #################  you do not need to add the feature maps of the conv1 layer
    block12 = layers.concatenate([conv1, block1, nn], axis=-1) #################  you do not need to use the feature maps of the attention block
    #pool2 = MaxPooling2D(pool_size=(2,2), name='pool2')(block1)
    conv2 = Conv2D(32, kernel_size=1, activation='relu', name='conv2', kernel_regularizer=regularizers.l2(0.0005))(block12)

# Graph-based global reasoning module 2 # input is 8x8x32

    graph_t2 = Reshape((32,WS*WS))(conv2)
    squeezed_graph_t2 = Conv1D(64, 1, activation='relu', name='squeezer2', kernel_regularizer=regularizers.l2(0.0001))(graph_t2)
    squeezed_graph2 = Reshape((64, 32))(squeezed_graph_t2)
    gconv2 = Conv1D(32, 1, activation='relu', name='gconv2', kernel_regularizer=regularizers.l2(0.0001))(squeezed_graph2)
    gconv_transpose2 = Reshape((64, 32))(gconv2)
    unsqueezed_graph_t2 = Conv1D(WS*WS, 1, activation='relu', name='unsqueezer2', kernel_regularizer=regularizers.l2(0.0001))(gconv_transpose2)
    unsqueezed_graph2 = Reshape((WS*WS, 64))(unsqueezed_graph_t2)
    glore2 = Reshape((WS, WS, 64))(unsqueezed_graph2)

    block2 = layers.concatenate([conv2, glore2], axis=-1) #################  you do not need to add the feature maps of the conv2 layer
    block21 = layers.concatenate([conv2, block2, nn], axis=-1) #################  you do not need to use the feature maps of the attention block2

    conv3 = Conv2D(32, kernel_size=1, activation='relu', name='conv3', kernel_regularizer=regularizers.l2(0.0005))(block21)
    conv4 = Conv2D(64, kernel_size=3, activation='relu', name='conv4', kernel_regularizer=regularizers.l2(0.0005))(conv3)
    conv5 = Conv2D(128, kernel_size=1, activation='relu', name='conv5', kernel_regularizer=regularizers.l2(0.0005))(conv4)

    flat = Flatten()(conv5)
    fc1 = Dense(100, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01))(flat)
    fc2 = Dense(20, activation='relu', name='fc2', kernel_regularizer=regularizers.l2(0.01))(fc1)
    fc3 = Dense(Num_Classes, activation='softmax', name='fc3', kernel_regularizer=regularizers.l2(0.0001))(fc2)

    model = Model(inputs=input_layer, outputs=fc3, name='CNN_model_with_global_reasoning_module')
    return model
# Attention GCN Ends

#Diff- Former Starts
class SWiGLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SWiGLU, self).__init__(**kwargs)

    def call(self, inputs):
        x, gate = tf.split(inputs, 2, axis=-1)
        return x * tf.sigmoid(gate) + x

def Positional_Encoding(max_len, d_emb):
    pos = tf.cast(tf.range(max_len)[:, tf.newaxis], tf.float32)
    i = tf.cast(tf.range(d_emb)[tf.newaxis, :], tf.float32)
    angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_emb, tf.float32))
    angles = pos * angle_rates
    sines = tf.sin(angles[:, 0::2])
    cosines = tf.cos(angles[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    return pos_encoding[tf.newaxis, ...]

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
            ffn_output = self.swiGLU1(ffn_output)
            ffn_output = self.dense2(ffn_output)
            ffn_output = self.dropout2(ffn_output, training=training)
            ffn_output = self.layernorm2(attn_output + ffn_output)
            x = ffn_output
        return x

def DSST(WS, k, Num_Classes, num_transformer_layers=2, num_heads=num_heads, dff=4 * 64, dropout_rate=0.1, layernorm_eps=1e-3):
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
    # pos_encoding = Positional_Encoding(num_patches + 1, embedding_dim)
    seq_len = tf.shape(patch_embed)[1]
    pos_encoding = Positional_Encoding(seq_len, embedding_dim)
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
# Diff-Former Ends

#Model Complexity Calculation
def count_model_parameters(model):
    """Count parameters using the same Keras API used by the GitHub baseline.

    model.count_params() is the canonical total. Trainable and non-trainable
    counts are reported separately so the result is auditable.
    """
    total = int(model.count_params())
    trainable = int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
    non_trainable = int(sum(np.prod(weight.shape) for weight in model.non_trainable_weights))

    if total != trainable + non_trainable:
        raise RuntimeError(
            "Keras parameter accounting mismatch: "
            f"total={total}, trainable={trainable}, non_trainable={non_trainable}."
        )

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def get_flops(model, input_shape):
    """Profile TensorFlow graph FLOPs for one input sample.

    This replaces the previous hand-written layer counter, which double-counted
    nested layers and missed/under-counted attention, Dense token positions,
    Conv3D operations, and raw TensorFlow matmuls.

    Returns None when TensorFlow cannot freeze/profile a model graph.
    """
    if len(input_shape) != 3:
        raise ValueError(f"Expected input_shape=(H, W, C), received {input_shape}.")

    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2,
        )

        input_spec = tf.TensorSpec(
            shape=(1, *tuple(input_shape)),
            dtype=tf.float32,
            name="model_input",
        )
        concrete_function = tf.function(
            lambda x: model(x, training=False)
        ).get_concrete_function(input_spec)
        frozen_function = convert_variables_to_constants_v2(concrete_function)

        options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        options["output"] = "none"
        profile = tf.compat.v1.profiler.profile(
            graph=frozen_function.graph,
            options=options,
        )
        return int(profile.total_float_ops) if profile is not None else None
    except Exception as exc:
        warnings.warn(f"FLOP profiling failed: {exc}")
        return None


def compute_flops(model, input_shape):
    return get_flops(model, input_shape)


# Train and Evaluate Model
def train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, output_dir, HSID, teRatio, k, WS, DLM, RDHSI, GT,Num_Classes,batch_size,epochs, target_names):
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(output_dir, "") + model_name.__name__
    os.makedirs(output_dir, exist_ok=True)
    print(model_name.__name__)
    input_shape = (WS, WS, k)  # Define the input shape for the model
    # if model_name == WaveMamba:
    #     Tr = Tr.reshape(-1, WS, WS, k)
    #     Va = Va.reshape(-1, WS, WS, k)
    #     Te = Te.reshape(-1, WS, WS, k)
    #     model = WaveMamba(Tr, batch_size, Num_Classes)
    if model_name == WaveFormer:
      # Transform Training/Validation/Test samples
      Tr = Tr.reshape(-1, WS, WS, k)
      Va = Va.reshape(-1, WS, WS, k)
      Te = Te.reshape(-1, WS, WS, k)
      Tr = wavelet_transform_image(Tr)
      Va = wavelet_transform_image(Va)
      Te = wavelet_transform_image(Te)
      model = WaveFormer(WS, k, Num_Classes)
    elif model_name == SSFormer:
      # Transform Training/Validation/Test samples
      Tr = Tr.reshape(-1, WS, WS, k)
      Va = Va.reshape(-1, WS, WS, k)
      Te = Te.reshape(-1, WS, WS, k)
      model = SSFormer(WS, k, Num_Classes)
    elif model_name == PyFormer:
      # Transform Training/Validation/Test samples
      Tr = Tr.reshape(-1, WS, WS, k)
      Va = Va.reshape(-1, WS, WS, k)
      Te = Te.reshape(-1, WS, WS, k)
      model = PyFormer(WS, k, Num_Classes)
    else:
        model = model_name(WS,k,Num_Classes)
    # Calling Custom Model
    #model = model_name(WS, k, Num_Classes)
    file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_model_summary_of_{model_name.__name__}.txt"
    # Assuming `model` is your Keras model
    total_params = model.count_params()
    parameter_stats = count_model_parameters(model)
    # flops = get_flops(model)
    flops = get_flops(model, input_shape=input_shape)
    with open(os.path.join(output_dir, file_name),'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        fh.write(f"\nTotal Parameters: {total_params}\n")
        fh.write(f"\nTotal Parameters: {parameter_stats['total']}\n")
        fh.write(f"Trainable Parameters: {parameter_stats['trainable']}\n")
        fh.write(f"Non-trainable Parameters: {parameter_stats['non_trainable']}\n")
        fh.write("FLOPs (batch=1): "
            + (str(flops) if flops is not None else "NA")
            + "\n"
        )
        
    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # Training the model
    start = time.time()
    history = model.fit(x=Tr, y=TrC, batch_size=batch_size, epochs=epochs, validation_data=(Va, VaC))
    end = time.time()
    Tr_Time = end - start
    # Predicting with the model
    start = time.time()
    Te_Pred = model.predict(Te)
    end = time.time()
    Te_Time = end - start
    ##Classification Report
    classification,confusion,oa,each_acc,aa,kappa = ClassificationReports(TeC, Te_Pred, target_names,  zero_division=0)
    print(classification)
    #     # Writing Results in CSV File
    file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_Classification_Report_{model_name.__name__}.csv"
    CSV_file = CSVResults(os.path.join(output_dir, file_name), classification, confusion, Tr_Time, Te_Time, DL_Time, kappa, oa, aa,each_acc)
    # #Confusion Matrix
    cm, annot = Conf_Mat(Te_Pred, TeC, target_names)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(cm, cmap= "Spectral", annot=annot, fmt='', ax=ax, linewidths=0.5)
    file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_Confusion_Matrix_{model_name.__name__}.png"
    plt.savefig(os.path.join(output_dir, file_name), dpi=500)
    # PLot the Ground Truth value
    # outputs = GT_Plot(CRDHSI, GT, model, WS, k)
    plt.figure(figsize=(10, 10))
    plt.imshow(GT, cmap='nipy_spectral')
    plt.axis('off')
    file_name = f"{HSID}_{trRatio}_{k}_{WS}_{DLM}_Ground_Truths.png"
    plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)
    # Ground Truths
    outputs = GT_Plot(CRDHSI, GT, model, WS, k)
    plt.figure(figsize=(10, 10))
    plt.imshow(outputs, cmap='nipy_spectral')
    plt.axis('off')
    file_name = f"{HSID}_{trRatio}_{k}_{WS}_{DLM}_Ground_Truths.png"
    plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)

    return history, Tr_Time,Te_Time, oa, aa,kappa, total_params, parameter_stats, flops

# # Define a list of model names
model_names = [WaveFormer] #[DSST] 
# Loop over the model names and run train_and_evaluate_model() for each model
all_results = {}
history_list = []
Tr_Time_list= []
Te_Time_list= []
for model_name in model_names:
    history, Tr_Time,Te_Time, oa, aa,kappa, total_params, parameter_stats, flops = train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, output_dir, HSID,
                                                                                    teRatio, k, WS, DLM, RDHSI, GT, Num_Classes,
                                                                                    batch_size, epochs, target_names)
    history_list.append(history)
    Tr_Time_list.append(Tr_Time)
    Tr_Time_list.append(Te_Time)

    # Add results to the dictionary
    model_function_name = model_name.__name__
    all_results[model_function_name] = {
        "history": history.history,
        "Tr_Time": Tr_Time,
        "Te_Time": Te_Time,
        "Overall Accuracy": oa,
        "Average Accuracy": aa,
        "Kappa": kappa,
        "Total Param": total_params,
        "Total Parameters": parameter_stats["total"],
        "Trainable Parameters": parameter_stats["trainable"],
        "Non-trainable Parameters": parameter_stats["non_trainable"],
        "FLOPs (batch=1)": flops,
    }

# Define the filename for all results
all_results_filename = "all_models_results.json"

# Concatenate output_dir with the filename
all_results_output_path = os.path.join(output_dir, all_results_filename)

# Convert the dictionary to JSON format
json_data = json.dumps(all_results, indent=4)

# Optionally, write the JSON data to a file
with open(all_results_output_path, 'w') as json_file:
    json_file.write(json_data)


# Define font size
font_size = 20

# Set font properties
font_prop = font_manager.FontProperties(size=font_size, weight='bold')

# Plot training and validation loss and accuracy for each model on one graph
fig, axs = plt.subplots(1, 2, figsize=(30, 20))
# colors = ['red', 'green', 'blue', 'purple', 'gray','black','brown','black','']
# linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
colors = ['red', 'green', 'blue', 'purple', 'cyan', 'orange', 
    'brown', 'pink', 'gray', 'maroon', 'gold']

linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), 
    (0, (5, 1)), (0, (5, 5)), (0, (1, 1)), 
    (0, (3, 5, 1, 5)), (0, (3, 10, 1, 10)), (0, (7, 3))]
for i, history in enumerate(history_list):
    # Plot loss
    axs[0].plot(history.history['loss'], label=f'{model_names[i].__name__} Train', color=colors[i], linestyle=linestyles[i], linewidth=3)
    axs[0].plot(history.history['val_loss'], label=f'{model_names[i].__name__} Val', color=colors[i], linestyle='--', linewidth=3)

    # Plot accuracy
    axs[1].plot(history.history['accuracy'], label=f'{model_names[i].__name__} Train', color=colors[i], linestyle=linestyles[i], linewidth=3)
    axs[1].plot(history.history['val_accuracy'], label=f'{model_names[i].__name__} Val', color=colors[i], linestyle='--', linewidth=3)

# Add labels and legend with increased font size
axs[0].set_title('Training and Validation Loss', fontproperties=font_prop, fontsize=24)
axs[0].set_xlabel('Epoch', fontproperties=font_prop, fontsize=20)
axs[0].set_ylabel('Loss', fontproperties=font_prop, fontsize=20)
axs[0].legend(prop=font_manager.FontProperties(size=18), loc='center right')

axs[1].set_title('Training and Validation Accuracy', fontproperties=font_prop, fontsize=24)
axs[1].set_xlabel('Epoch', fontproperties=font_prop, fontsize=20)
axs[1].set_ylabel('Accuracy', fontproperties=font_prop, fontsize=20)
axs[1].legend(prop=font_manager.FontProperties(size=18), loc='center right')

# Increase tick label size
axs[0].tick_params(labelsize=font_size)
axs[1].tick_params(labelsize=font_size)

# Add grid lines and set background color
axs[0].grid(True)
axs[1].grid(True)
fig.patch.set_facecolor('#f2f2f2')
plt.tight_layout()

# Save the figure
file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_acc_loss_curve_all_models.png"
plt.savefig(os.path.join(output_dir, file_name), dpi=500)
tf.keras.backend.clear_session()
