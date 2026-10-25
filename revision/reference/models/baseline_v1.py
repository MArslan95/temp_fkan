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
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Input, AveragePooling2D, Activation, MaxPooling2D, BatchNormalization, Concatenate
from keras.layers import (Conv3D, Dense, Dropout,Input, Reshape, Reshape,Activation,LayerNormalization)
from tensorflow.keras.layers import (Input, InputSpec, Add, AveragePooling2D, LeakyReLU, Multiply, 
                                     Conv3D, Conv2D, Conv1D, BatchNormalization,
                                     GlobalAveragePooling2D, GlobalMaxPooling2D, DepthwiseConv2D, Layer, Dense, 
                                     Layer, Dropout, Activation, Flatten, Reshape, Concatenate, MaxPooling2D)
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
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Script Description")
# Add arguments for each parameter
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    choices=["LK", "HC", "UH13", "PU", "PC", "SA","HH","QUH","PI","TH","Fe"],
    default="PI",
    help="Dataset identifier",
)
parser.add_argument("--window_size", "-ws", type=int, default=4, help="Window size")    
parser.add_argument("--dl_method", "-dlm", type=str, default='PCA', help="Dimension Reduction Method")  # 차원 축소 메소드
parser.add_argument("--test_ratio", "-te", type=float, default=0.50, help="Test ratio")
parser.add_argument("--train_ratio", "-tr", type=float, default=0.99, help="Train ratio")
parser.add_argument(
    "--validation_ratio", "-vr", type=float, default=0.01, help="Validation ratio"
)
parser.add_argument(
    "--number_of_bands",
    "-k",
    type=int,
    default=15,
    help="Number of bands principal components for dimensionality reduction",
)
parser.add_argument(
    "--epochs", "-e", type=int, default=50, help="Number of epochs for training"
)
parser.add_argument(
    "--batch_size", "-bs", type=int, default=56, help="Batch size for training"
)
# Parse the command-line arguments
args = parser.parse_args()

# DataSet Loader Funtion
base_dir = '../hsi_datasets'
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
        GT = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_corrected']
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
output_dir = os.path.join(f"baseline_res_{out_channels}_{num_heads}_{batch_size}/{HSID}/{teRatio}/{WS}")
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


# Flops Calculation
def get_flops(model, input_shape=None):
    total_flops = 0

    # Build model with dummy input to infer shapes if input_shape is provided
    if input_shape and isinstance(model, tf.keras.Model) and not model.built:
        dummy_input = tf.keras.Input(shape=input_shape)
        model(dummy_input)
    
    def process_layer(layer, input_shape=None):
        nonlocal total_flops
        flops = 0

        # Helper to get input/output shapes
        def get_shapes(layer):
            in_shape = layer.input_shape if hasattr(layer, 'input_shape') else input_shape
            out_shape = layer.output_shape if hasattr(layer, 'output_shape') else None
            return in_shape, out_shape

        # Handle Conv2D layers
        if isinstance(layer, tf.keras.layers.Conv2D):
            in_shape, out_shape = get_shapes(layer)
            if in_shape and out_shape:
                input_channels = in_shape[-1]
                output_height = out_shape[1]
                output_width = out_shape[2]
                kernel_size = layer.kernel_size[0] * layer.kernel_size[1] * input_channels
                flops += kernel_size * output_height * output_width * layer.filters * 2
                if layer.activation is not None and layer.activation.__name__ != 'linear':
                    flops += output_height * output_width * layer.filters

        # Handle Conv3D layers
        elif isinstance(layer, tf.keras.layers.Conv3D):
            in_shape, out_shape = get_shapes(layer)
            if in_shape and out_shape:
                input_channels = in_shape[-1]
                output_depth = out_shape[1]
                output_height = out_shape[2]
                output_width = out_shape[3]
                kernel_size = layer.kernel_size[0] * layer.kernel_size[1] * layer.kernel_size[2] * input_channels
                flops += kernel_size * output_depth * output_height * output_width * layer.filters * 2
                if layer.activation is not None and layer.activation.__name__ != 'linear':
                    flops += output_depth * output_height * output_width * layer.filters

        # Handle Dense layers
        elif isinstance(layer, tf.keras.layers.Dense):
            in_shape, out_shape = get_shapes(layer)
            input_dim = layer.get_weights()[0].shape[0] if layer.get_weights() else in_shape[-1] if in_shape else 0
            output_dim = layer.units
            flops += 2 * input_dim * output_dim  # Multiplications + additions
            if layer.activation is not None and layer.activation.__name__ != 'linear':
                flops += output_dim  # Approximate activation cost

        # AttenctionGCN Flops
         # Handle Reshape layers
        elif isinstance(layer, tf.keras.layers.Reshape):
            flops += 0  # Reshape is just memory rearrangement, negligible FLOPs
        # Handle Flatten layers
        elif isinstance(layer, tf.keras.layers.Flatten):
            flops += 0  # Flatten is just memory rearrangement, negligible FLOPs
        # Handle concatenate layers
        elif isinstance(layer, tf.keras.layers.Concatenate):
            flops += 0  # Concatenate is just memory copying, negligible FLOPs
        # Handle cot_attention (custom attention layer)
        elif layer.__class__.__name__ == 'cot_attention':  # Assuming cot_attention is a custom layer
            in_shape, out_shape = get_shapes(layer)
            if in_shape and out_shape:
                height, width, channels = in_shape[1], in_shape[2], in_shape[3]
                # Assume cot_attention uses a 3x3 convolution to compute attention scores
                kernel_size = 3 * 3 * channels  # 3x3 kernel
                attention_flops = kernel_size * height * width * channels * 2  # Conv for attention scores
                # Softmax approximation
                attention_flops += height * width * channels  # One operation per element
                # Weighted sum (attention weights * input)
                attention_flops += 2 * height * width * channels * height * width  # Matrix multiplication
                flops += attention_flops
        # Handle LayerNormalization
        elif isinstance(layer, tf.keras.layers.LayerNormalization):
            in_shape, _ = get_shapes(layer)
            if in_shape:
                flops += 5 * np.prod(in_shape[1:])  # Mean, variance, scale, shift, normalize
        # Handle Dropout (negligible FLOPs)
        elif isinstance(layer, tf.keras.layers.Dropout):
            flops += 0
        # Handle functional model operations
        elif isinstance(layer, tf.keras.Model):
            for sublayer in layer.layers:
                if sublayer.name.startswith('concatenate'):
                    flops += 0  # Negligible
                else:
                    flops += process_layer(sublayer, input_shape)

        # Handle MultiHeadAttention layers
        elif isinstance(layer, tf.keras.layers.MultiHeadAttention):
            in_shape, _ = get_shapes(layer)
            embed_dim = layer._key_dim * layer._num_heads  # Use _num_heads
            seq_len = in_shape[-2] if in_shape and len(in_shape) > 2 else 0
            if seq_len:
                # Q, K, V projections
                flops += 3 * (2 * embed_dim * embed_dim * seq_len)  # Three dense layers
                # Scaled dot-product attention
                flops += 2 * layer._num_heads * seq_len * seq_len * (embed_dim // layer._num_heads)  # QK^T matmul
                flops += layer._num_heads * seq_len * seq_len  # Softmax (approx)
                flops += 2 * layer._num_heads * seq_len * seq_len * (embed_dim // layer._num_heads)  # Attention * V
                # Output projection
                flops += 2 * embed_dim * embed_dim * seq_len

        # Handle Transformer/Transformer_SSViT layers
        elif isinstance(layer, (Transformer, Transformer_SSViT)):
            for _ in range(layer.transformer_layer_depth):
                flops += process_layer(layer.mha, input_shape)
                flops += process_layer(layer.dropout1, input_shape)
                flops += process_layer(layer.layernorm1, input_shape)
                flops += process_layer(layer.dense1, input_shape)
                flops += process_layer(layer.dense2, input_shape)
                flops += process_layer(layer.dropout2, input_shape)
                flops += process_layer(layer.layernorm2, input_shape)

        # Handle LayerNormalization
        elif isinstance(layer, tf.keras.layers.LayerNormalization):
            in_shape, _ = get_shapes(layer)
            if in_shape:
                flops += 5 * np.prod(in_shape[1:])  # Mean, variance, scale, shift, normalize

        # Handle Dropout (negligible FLOPs)
        elif isinstance(layer, tf.keras.layers.Dropout):
            flops += 0

        # Handle MulMultiHeadAttention
        elif isinstance(layer, MulMultiHeadAttention):
            flops += process_layer(layer.query)  # Dense layer
            flops += process_layer(layer.key)    # Dense layer
            flops += process_layer(layer.value)  # Dense layer
            flops += process_layer(layer.dropout)  # Dropout layer (typically 0 FLOPs)
            # Add FLOPs for attention computation
            batch_size = 1  # Assume batch_size=1 for simplicity; adjust if known
            seq_len = layer.embed_dim // layer.num_heads  # Approximation
            flops += (2 * batch_size * layer.num_heads * seq_len * seq_len)  # QK^T matmul
            flops += (batch_size * layer.num_heads * seq_len * seq_len)      # Softmax (approx.)
            flops += (2 * batch_size * layer.num_heads * seq_len * seq_len)
            
        # Handle MulSpectralSpatialFeatureEnhancement
        elif isinstance(layer, MulSpectralSpatialFeatureEnhancement):
            flops += process_layer(layer.spatial_gate, input_shape)
            flops += process_layer(layer.spectral_gate, input_shape)
            in_shape, _ = get_shapes(layer)
            if in_shape:
                seq_len = in_shape[1] if len(in_shape) > 1 else 1
                out_channels = layer.spatial_gate.layers[0].units
                flops += 2 * in_shape[0] * seq_len * out_channels  # spatial_tokens * gate, spectral_tokens * gate
         # Handle MulStateSpaceModel
        elif isinstance(layer, MulStateSpaceModel):
            flops += process_layer(layer.state_transition, input_shape)
            flops += process_layer(layer.state_update, input_shape)
            in_shape, _ = get_shapes(layer)
            if in_shape:
                seq_len = in_shape[1] if len(in_shape) > 1 else 1
                flops += seq_len * (flops + in_shape[0] * layer.state_dim)  # State updates per timestep

        # Handle WaveSpectralSpatialFeatureEnhancement
        elif isinstance(layer, WaveSpectralSpatialFeatureEnhancement):
            flops += process_layer(layer.spatial_gate, input_shape)
            flops += process_layer(layer.spectral_gate, input_shape)
            in_shape, _ = get_shapes(layer)
            if in_shape:
                seq_len = in_shape[1] if len(in_shape) > 1 else 1
                out_channels = layer.spatial_gate.layers[0].units
                flops += 2 * in_shape[0] * seq_len * out_channels  # spatial_tokens * gate, spectral_tokens * gate
        # Handle WaveStateSpaceModel
        elif isinstance(layer, WaveStateSpaceModel):
            flops += process_layer(layer.state_transition, input_shape)
            flops += process_layer(layer.state_update, input_shape)
            in_shape, _ = get_shapes(layer)
            if in_shape:
                seq_len = in_shape[1] if len(in_shape) > 1 else 1
                flops += seq_len * (flops + in_shape[0] * layer.state_dim)  # State updates per timestep

        # Handle WaveSSMambaModel
        elif isinstance(layer, WaveSSMambaModel):
            flops += process_layer(layer.token_generation, input_shape)
            flops += process_layer(layer.feature_enhancement, input_shape)
            in_shape, _ = get_shapes(layer)
            if in_shape:
                batch_size = in_shape[0]
                height = in_shape[1]
                width = in_shape[2]
                channels = in_shape[3] if len(in_shape) > 3 else 1
                wavelet_flops_per_pixel = 8
                flops_per_transform = wavelet_flops_per_pixel * height * width * channels
                num_tokens = height
                flops += 2 * batch_size * num_tokens * flops_per_transform  # 2x for spatial and spectral
            flops += process_layer(layer.state_space_model, input_shape)
            flops += process_layer(layer.dense, input_shape)
            flops += process_layer(layer.dropout, input_shape)
            flops += process_layer(layer.classifier, input_shape)

        # Handle SSMambaModel
        elif isinstance(layer, SSMambaModel):
            flops += process_layer(layer.token_generation, input_shape)
            flops += process_layer(layer.feature_enhancement, input_shape)
            in_shape, _ = get_shapes(layer)
            if in_shape:
                batch_size = in_shape[0]
                height = in_shape[1]
                width = in_shape[2]
                channels = in_shape[3] if len(in_shape) > 3 else 1
                wavelet_flops_per_pixel = 8
                flops_per_transform = wavelet_flops_per_pixel * height * width * channels
                num_tokens = height
                flops += 2 * batch_size * num_tokens * flops_per_transform  # 2x for spatial and spectral
            flops += process_layer(layer.state_space_model, input_shape)
            flops += process_layer(layer.dense, input_shape)
            flops += process_layer(layer.dropout, input_shape)
            flops += process_layer(layer.classifier, input_shape)
            
        elif isinstance(layer, SSMSpectralSpatialFeatureEnhancement):
            flops += process_layer(layer.spatial_gate, input_shape)
            flops += process_layer(layer.spectral_gate, input_shape)
            in_shape, _ = get_shapes(layer)
            if in_shape:
                seq_len = in_shape[1] if len(in_shape) > 1 else 1
                out_channels = layer.spatial_gate.layers[0].units
                flops += 2 * in_shape[0] * seq_len * out_channels
        # Handle functional model operations
        elif isinstance(layer, tf.keras.Model):
            for sublayer in layer.layers:
                if sublayer.name.startswith('add'):  # Positional encoding addition
                    in_shape = sublayer.input_shape[0] if sublayer.input_shape else None
                    if in_shape:
                        flops += np.prod(in_shape[1:])  # Element-wise addition
                elif sublayer.name.startswith('matmul'):  # Cross-attention matmul
                    in_shape_1 = sublayer.input_shape[0] if sublayer.input_shape else None
                    in_shape_2 = sublayer.input_shape[1] if sublayer.input_shape and len(sublayer.input_shape) > 1 else None
                    if in_shape_1 and in_shape_2:
                        flops += 2 * in_shape_1[-1] * in_shape_2[-1] * in_shape_1[-2]  # Matmul
                elif sublayer.name.startswith('softmax'):  # Cross-attention softmax
                    in_shape = sublayer.input_shape if sublayer.input_shape else None
                    if in_shape:
                        flops += np.prod(in_shape[1:])  # Approximate softmax
                elif sublayer.name.startswith('concatenate'):  # Concatenation
                    flops += 0  # Negligible
                elif sublayer.name.startswith('expand_dims'):  # tf.expand_dims
                    flops += 0  # Negligible
                else:
                    flops += process_layer(sublayer, input_shape)

        # Handle Sequential or nested models
        elif isinstance(layer, tf.keras.Sequential) or hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                flops += process_layer(sublayer, input_shape)
      
        total_flops += flops
        print(f"Layer {layer.name}: {flops:,} FLOPs")
        return flops

    process_layer(model)
    return total_flops

def compute_flops(model):
    return get_flops(model)   

# Train and Evaluate Model
def train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, output_dir, HSID, teRatio, k, WS, DLM, RDHSI, GT,Num_Classes,batch_size,epochs, target_names):
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(output_dir, "") + model_name.__name__
    os.makedirs(output_dir, exist_ok=True)
    print(model_name.__name__)
    input_shape = (WS, WS, k)  # Define the input shape for the model
    if model_name == WaveMamba:
        Tr = Tr.reshape(-1, WS, WS, k)
        Va = Va.reshape(-1, WS, WS, k)
        Te = Te.reshape(-1, WS, WS, k)
        model = WaveMamba(Tr, batch_size, Num_Classes)
    elif model_name == MulMamba:
        model = MulMamba(Tr, batch_size)
    elif model_name == SSM:
        model = SSM(Tr, batch_size)
    elif model_name == SSMamba:
        model = SSMamba(Tr, batch_size)
    else:
        model = model_name(WS,k,Num_Classes)
    # Calling Custom Model
    #model = model_name(WS, k, Num_Classes)
    file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_model_summary_of_{model_name.__name__}.txt"
    # Assuming `model` is your Keras model
    total_params = model.count_params()
    # flops = get_flops(model)
    flops = get_flops(model, input_shape=input_shape)
    with open(os.path.join(output_dir, file_name),'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        fh.write(f"\nTotal Parameters: {total_params}\n")
        fh.write(f"Total FLOPs: {flops}\n")
        
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

    return history, Tr_Time,Te_Time, oa, aa,kappa, total_params

# # Define a list of model names
model_names = [SSMamba] #[DSST] 
# Loop over the model names and run train_and_evaluate_model() for each model
all_results = {}
history_list = []
Tr_Time_list= []
Te_Time_list= []
for model_name in model_names:
    history, Tr_Time,Te_Time, oa, aa,kappa, total_params = train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, output_dir, HSID, teRatio, k, WS, DLM, RDHSI, GT, Num_Classes, batch_size, epochs, target_names)
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
        "Training Parameters": total_params,
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














































# #!/usr/bin/env python
# # coding: utf-8
# import os,argparse, json
# # os.environ["CUDA_VISIBLE_DEVICES"]="1"
# import time
# import warnings
# import math
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import scipy.io as sio
# import seaborn as sns
# import spectral
# import spectral.io.envi as envi

# import tensorflow as tf
# from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
# print(tf.config.list_physical_devices('GPU'))
# import keras
# import pywt
# from tensorflow.keras import layers, Model
# from tensorflow.keras.layers import  Conv1D, MaxPooling1D, Flatten, Reshape
# from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Input, AveragePooling2D, Activation, MaxPooling2D, BatchNormalization, Concatenate
# from keras.layers import (Conv3D, Dense, Dropout,Input, Reshape, Reshape,Activation,LayerNormalization)
# from keras.losses import categorical_crossentropy
# from tensorflow.keras import regularizers
# from keras.models import Model, Sequential
# from sklearn.manifold import TSNE
# # from vit_keras import utils, vit
# from keras.utils import to_categorical
# from keras_cv_attention_models import attention_layers
# import matplotlib.font_manager as font_manager
# from utils import TrTeSplit,DLMethod,ClassificationReports,Conf_Mat,CSVResults,GT_Plot
# import warnings
# warnings.filterwarnings('ignore')
# # Create an ArgumentParser object
# parser = argparse.ArgumentParser(description="Script Description")

# # Add arguments for each parameter
# parser.add_argument(
#     "--dataset",
#     "-d",
#     type=str,
#     choices= ["LK", "HC", "UH13", "PU", "PC", "SA","HH","QUH","PI","TH"], #["LK", "HC", "UH13", "IP", "BS", "KSC", "PU", "PC", "SA", "SLA"],
#     default="PU",
#     help="Dataset identifier",
# )
# parser.add_argument("--window_size", "-ws", type=int, default=4, help="Window size")
# parser.add_argument("--dl_method", "-dlm", type=str, default='PCA', help="Dimension Reduction Method")
# parser.add_argument("--test_ratio", "-te", type=float, default=0.50, help="Test ratio")
# parser.add_argument("--train_ratio", "-tr", type=float, default=0.99, help="Train ratio")
# parser.add_argument(
#     "--validation_ratio", "-vr", type=float, default=0.01, help="Validation ratio"
# )
# parser.add_argument(
#     "--number_of_bands",
#     "-k",
#     type=int,
#     default=15,
#     help="Number of bands principal components for dimensionality reduction",
# )
# parser.add_argument(
#     "--epochs", "-e", type=int, default=50, help="Number of epochs for training"
# )
# parser.add_argument(
#     "--batch_size", "-bs", type=int, default=56, help="Batch size for training"
# )

# # Parse the command-line arguments
# args = parser.parse_args()

# ## Loading Hyperspectral Datasets
# base_dir = './datasets' 
# def LoadHSIData(method):
#     data_path = os.path.join(os.getcwd(),f'{base_dir}')
#     if method == 'LK':
#         ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
#         HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
#         GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt'))['WHU_Hi_LongKou_gt']
#         Num_Classes = 9
#         target_names = ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',
#                         'Narrow-leaf soybean', 'Rice', 'Water',
#                         'Roads and houses', 'Mixed weed']
        
#     elif method == 'HH':
#         ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
#         HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
#         GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_gt'))['WHU_Hi_HongHu_gt']
#         Num_Classes = 22
#         target_names = ['Red roof', 'Road', 'Bare soil', 'Cotton',
#                         'Cotton firewood', 'Rape', 'Chinese cabbage',
#                         'Pakchoi', 'Cabbage', 'Tuber mustard', 'Brassica parachinensis',
#                         'Brassica chinensis', 'Small Brassica chinensis', 'Lactuca sativa',
#                         'Celtuce', 'Film covered lettuce', 'Romaine lettuce',
#                         'Carrot', 'White radish', 'Garlic sprout', 'Broad bean',
#                         'Tree']
        
#     elif method == 'HC':
#         ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
#         HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
#         GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_gt'))['WHU_Hi_HanChuan_gt']
#         Num_Classes = 16
#         target_names = ['Strawberry', 'Cowpea', 'Soybean', 'Sorghum',
#                         'Water spinach', 'Watermelon', 'Greens', 'Trees', 'Grass',
#                         'Red roof', 'Gray roof', 'Plastic', 'Bare soil', 'Road',
#                         'Bright object', 'Water']

#     elif method == 'IP':
#         HSI = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
#         GT = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
#         Num_Classes = 16
#         target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
#                         'Grass-pasture', 'Grass-trees', 'Grass-mowed',
#                         'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
#                         'Soybean-clean', 'Wheat', 'Woods', 'Buildings',
#                         'Stone-Steel']
        
#     elif method == 'BS':
#         HSI = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
#         GT = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
#         Num_Classes = 14
#         target_names = ['Water', 'Hippo Grass', 'Floodplain Grasses 1', 'Floodplain Grasses 2',
#                         'Reeds 1', 'Riparian', 'Firescar 2', 'Island Interior', 'Woodlands',
#                         'Acacia Shrublands', 'Acacia Grasslands', 'Short Mopane', 'Mixed Mopane', 'Exposed Soils']

#     elif method == 'KSC':
#         HSI = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
#         GT = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
#         Num_Classes = 13
#         target_names = ['Scrub', 'Willow Swamp', 'CP/Oak', 'CP hammock', 'Slash Pine', 'Oak/Broadleaf', 'Hardwood Swamp',
#                         'Graminoid Marsh', 'Spartina Marsh', 'Cattail Marsh', 'Salt Marsh', 'Mud Flats', 'Water']
     
#     elif method == 'PU':
#         HSI = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
#         GT = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
#         Num_Classes = 9
#         target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted','Soil','Bitumen',
#                         'Bricks','Shadows']

#     elif method == 'PC':
#         HSI = sio.loadmat(os.path.join(data_path, 'Pavia.mat'))['pavia']
#         GT = sio.loadmat(os.path.join(data_path, 'Pavia_gt.mat'))['pavia_gt']
#         Num_Classes = 9
#         target_names = ['Water', 'Trees', 'Asphalt', 'Bricks', 'Bitumen', 'Tiles', 'Shadows',
#                         'Meadows', 'Soil']

#     elif method == 'SA':
#         HSI = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
#         GT = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
#         Num_Classes = 16
#         target_names = ['Weeds_1','Weeds_2','Fallow',
#                         'Fallow_rough_plow','Fallow_smooth', 'Stubble','Celery',
#                         'Grapes_untrained','Soil_vinyard_develop','Corn_Weeds',
#                         'Lettuce_4wk','Lettuce_5wk','Lettuce_6wk',
#                         'Lettuce_7wk', 'Vinyard_untrained','Vinyard_trellis']
      
#     elif method == 'SLA':
#         HSI = sio.loadmat(os.path.join(data_path, 'SalinasA_corrected.mat'))['salinasA_corrected']
#         GT = sio.loadmat(os.path.join(data_path, 'SalinasA_gt.mat'))['salinasA_gt']
#         val_old = np.array([0,1,10,11,12,13,14])
#         val_new = np.array([0,1,2,3,4,5,6])
#         index = np.digitize(GT.ravel(), val_old, right=True)
#         GT = val_new[index].reshape(GT.shape)
#         Num_Classes = 6
#         target_names = ['Brocoli 1', 'Corn weeds', 'Lettuce 4wk', 'Lettuce 5wk',
#                        'Lettuce 6wk', 'Lettuce 7wk']

#     elif method == 'UH13':
#         HSI = sio.loadmat(os.path.join(data_path, 'HU13.mat'))['HSI']
#         GT = sio.loadmat(os.path.join(data_path, 'HU13_gt.mat'))['gt']
#         Num_Classes = 15
#         target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees',
#                     'Soil', 'Water', 'Residential', 'Commercial', 'Road',
#                     'Highway', 'Railway', 'Parking Lot 1', 'Parking Lot 2',
#                     'Tennis Court', 'Running Track']

#     elif method == 'QUH':
#         HSI = sio.loadmat(os.path.join(data_path,'QUH-Qingyun.mat'))['Chengqu']
#         GT = sio.loadmat(os.path.join(data_path,'QUH-Qingyun_GT.mat'))['ChengquGT']
#         Num_Classes = 6
#         target_names = ["Trees", "Concrete building", "Car", "Ironhide building", "Plastic playground", "Asphalt road"]
        
#     elif method == 'PI':
#         HSI = sio.loadmat(os.path.join(data_path,'QUH-Pingan.mat'))['Haigang']
#         GT = sio.loadmat(os.path.join(data_path,'QUH-Pingan_GT.mat'))['HaigangGT']
#         Num_Classes = 10
#         target_names = ["Ship", "Seawater", "Trees"," Concrete structure building", "Floating pier", "Brick houses", 
#                       "Steel houses"," Wharf construction land", "Car", "Road"]
#     elif method == 'TH':
#         HSI = sio.loadmat(os.path.join(data_path, 'QUH-Tangdaowan.mat'))['Tangdaowan']
#         GT = sio.loadmat(os.path.join(data_path, 'QUH-Tangdaowan_GT.mat'))['TangdaowanGT']
#         Num_Classes = 18
#         target_names = ['Rubber track', 'Flaggingv', 'Sandy', 'Asphalt', 'Boardwalk', 'Rocky shallows',
#             'Grassland', 'Bulrush', 'Gravel road', 'Ligustrum vicaryi', 'Coniferous pine',
#             'Spiraea', 'Bare soil', 'Buxus sinica', 'Photinia serrulata', 'Populus',
#             'Ulmus pumila L', 'Seawater']
#     else:
#         print("Wrong Choice")

#     return HSI, GT, Num_Classes, target_names

# ## Global Parameters for Loop
# HSID = args.dataset ## "SLA", "IP", "PU", "PC", "SA", "KSC", "BS", "LK", "HH" (difficult to compile), "HC"
# DLM = args.dl_method ## "PCA", "iPCA"
# WS = args.window_size
# # Using the calculated values
# teRatio = args.test_ratio
# trRatio = args.train_ratio
# vrRatio = args.validation_ratio
# k = args.number_of_bands
# adam = tf.keras.optimizers.legacy.Adam(lr = 0.001, decay = 1e-06)
# epochs = args.epochs
# batch_size = args.batch_size
# num_heads=8
# out_channels=32
# #output_dir = os.path.join(f"baseline_results_out_32_{num_heads}_{batch_size}/{HSID}/{teRatio}/{WS}")
# output_dir = os.path.join(f"baseline_PyFormer_{out_channels}_{num_heads}_{batch_size}/{HSID}/{teRatio}/{WS}")
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# ## Creat Patches for 3D (Spatial-Spectral) Models
# def ImageCubes(HSI, GT, WS=WS, removeZeroLabels=True):
#     num_rows, num_cols, num_bands = HSI.shape
#     margin = int(WS / 2)
#     padded_data = np.pad(HSI, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
#     image_cubes = np.zeros((num_rows * num_cols, WS, WS, num_bands))
#     patchesLabels = np.zeros((num_rows * num_cols))
#     patchIndex = 0
#     for r in range(margin, num_rows + margin):
#         for c in range(margin, num_cols + margin):
#             cube = padded_data[r - margin: r + margin, c - margin: c + margin, :]
#             image_cubes[patchIndex, :, :, :] = cube
#             patchesLabels[patchIndex] = GT[r-margin, c-margin]
#             patchIndex = patchIndex + 1
#     if removeZeroLabels:
#       image_cubes = image_cubes[patchesLabels>0,:,:,:]
#       patchesLabels = patchesLabels[patchesLabels>0]
#       patchesLabels -= 1
#     return image_cubes, patchesLabels



# # **Loading HSI and Creat Patchs**
# # Main Function to load Datasets, Dimensional Reduction and Creating Patchs for CNN
# HSI, GT, Num_Classes, target_names = LoadHSIData(HSID)
# ## Reduce the Dimensionality
# # if k < HSI.shape[2]:
# start = time.time()
# RDHSI = DLMethod(DLM, HSI, NC = k)
# end = time.time()
# DL_Time = end - start
# ## Create Image Cubes for Model Building
# CRDHSI, CGT = ImageCubes(RDHSI, GT, WS = WS)
# ## Split Train/validation and Test sets
# Tr, Va, Te, TrC, VaC, TeC = TrTeSplit(CRDHSI, CGT, trRatio, vrRatio, teRatio)
# # Reshape train, validation, and test sets
# # Tr = Tr.reshape(-1, WS, WS, k, 1)
# TrC = to_categorical(TrC)
# # Va = Va.reshape(-1, WS, WS, k, 1)
# VaC = to_categorical(VaC)
# # Te = Te.reshape(-1, WS, WS, k, 1)
# TeC = to_categorical(TeC)

# # Hybrid Vision Transformer Starts
# # Positional Encoding
# def get_positional_encoding(max_len, d_emb):
#     pos = np.arange(max_len)[:, np.newaxis]
#     i = np.arange(d_emb)[np.newaxis, :]
#     angles = pos / np.power(10000, 2 * i / d_emb)
#     positional_encoding = np.zeros((max_len, d_emb))
#     positional_encoding[:, ::2] = np.sin(angles[:, ::2])
#     positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])
#     return positional_encoding[np.newaxis, ...]

# class Transformer(tf.keras.layers.Layer):
#     def __init__(self, num_heads, transformer_layer_depth, d_model, dff, dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
#         super(Transformer, self).__init__(**kwargs)
#         self.num_heads = num_heads
#         self.transformer_layer_depth = transformer_layer_depth
#         self.d_model = d_model
#         self.dff = dff
#         self.dropout_rate = dropout_rate
#         self.layernorm_eps = layernorm_eps

#         self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
#         self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
#         self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(d_model)
#         self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

#     def call(self, inputs, training):
#         x = inputs
#         for i in range(self.transformer_layer_depth):
#             # Multi-Head Attention
#             attn_output = self.mha(x, x)
#             attn_output = self.dropout1(attn_output, training=training)
#             attn_output = self.layernorm1(x + attn_output)
#             # Feed Forward Network
#             ffn_output = self.dense1(attn_output)
#             ffn_output = self.dense2(ffn_output)
#             ffn_output = self.dropout2(ffn_output, training=training)
#             ffn_output = self.layernorm2(attn_output + ffn_output)
#             x = ffn_output
#         return x

# def Hybrid_ViT(WS, k, Num_Classes, num_transformer_layers=4, num_heads=8, dff=4*64, dropout_rate=0.1, layernorm_eps=1e-3):
#     input_shape = (WS, WS, k)

#     max_patch_size = min(input_shape[:3])
#     patch_size = max_patch_size
#     num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size) * (input_shape[2] // patch_size)
#     embedding_dim = 64
#     inputs = Input(shape=input_shape, name='inputs')
#     x = tf.expand_dims(inputs, axis=1) # add this line
#     # Create a patch embedding layer
#     # patch_embed = Conv3D(embedding_dim, kernel_size=(1, patch_size,  ), strides=(1, patch_size, patch_size), padding='valid', name='patch_embed')(x)
#     patch_embed = Conv3D(embedding_dim, kernel_size=(1, patch_size, patch_size), strides=(1, patch_size, patch_size), padding='valid', name='patch_embed')(x)
#     cls_token = tf.Variable(tf.zeros((1, 1, 1, embedding_dim)), name='cls_token')
#     # Add cls_token to patch embeddings
#     patch_embed = tf.concat([tf.broadcast_to(cls_token, [tf.shape(patch_embed)[0], 1, 1, 1, embedding_dim]), patch_embed], axis=1)
#     # Add positional encoding to patch embeddings
#     pos_encoding = get_positional_encoding(num_patches+1, embedding_dim)

#     patch_embed = patch_embed + pos_encoding
#     # Create a Transformer encoder
#     for i in range(num_transformer_layers):
#         if i == 0:
#             transformer_layer = Transformer(num_heads=num_heads, transformer_layer_depth=1, d_model=embedding_dim, dff=dff, name=f'transformer_layer_{i}')(patch_embed)
#         else:
#             transformer_layer = Transformer(num_heads=num_heads, transformer_layer_depth=1, d_model=embedding_dim, dff=dff, name=f'transformer_layer_{i}')(transformer_layer)

#     # Use cross-attention between the patch embeddings and a separate set of learnable embeddings
#     q = Dense(embedding_dim)(transformer_layer[:, 0, 0, 0])
#     k = Dense(embedding_dim)(patch_embed[:, 1:, :, :, :])
#     v = patch_embed[:, 1:, :, :, :]
#     cross_attn_output = tf.matmul(q[:, tf.newaxis, :], k, transpose_b=True)
#     cross_attn_output = cross_attn_output / tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))
#     cross_attn_output = tf.nn.softmax(cross_attn_output, axis=-1)
#     cross_attn_output = tf.matmul(cross_attn_output, v)
#     # Concatenate cls_token and cross-attention output
#     output_layer = tf.concat([transformer_layer[:, 0, 0, 0], cross_attn_output[:, 0, 0, 0]], axis=-1)
#     # Add a few dense layers and a final softmax layer
#     output_layer = Dense(dff, activation='relu')(output_layer)
#     output_layer = Dense(dff, activation='relu')(output_layer)
#     output_layer = Dense(Num_Classes, activation='softmax', name='output_layer')(output_layer)
#     # Define the model
#     model = Model(inputs=inputs, outputs=output_layer, name='Hybrid_ViT')
#     return model
# # Hybrid Vision Transformer Ends

# # SSViT Starts
# def get_positional_encoding_ssvit(max_len, d_emb):
#     pos = np.arange(max_len)[:, np.newaxis]
#     i = np.arange(d_emb)[np.newaxis, :]
#     angles = pos / np.power(10000, 2 * i / d_emb)
#     positional_encoding = np.zeros((max_len, d_emb))
#     positional_encoding[:, ::2] = np.sin(angles[:, ::2])
#     positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])
#     return positional_encoding[np.newaxis, ...]


# class Transformer_SSViT(tf.keras.layers.Layer):
#     def __init__(
#         self,
#         num_heads,
#         transformer_layer_depth,
#         d_model,
#         dff,
#         dropout_rate=0.1,
#         layernorm_eps=1e-6,
#         **kwargs,
#     ):
#         super(Transformer_SSViT, self).__init__(**kwargs)
#         self.num_heads = num_heads
#         self.transformer_layer_depth = transformer_layer_depth
#         self.d_model = d_model
#         self.dff = dff
#         self.dropout_rate = dropout_rate
#         self.layernorm_eps = layernorm_eps

#         self.mha = tf.keras.layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=d_model
#         )
#         self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
#         self.dense1 = tf.keras.layers.Dense(dff, activation="relu")
#         self.dense2 = tf.keras.layers.Dense(d_model)
#         self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

#     def call(self, inputs, training):
#         x = inputs
#         for i in range(self.transformer_layer_depth):
#             # MultiHead Attention
#             attn_output = self.mha(x, x)
#             attn_output = self.dropout1(attn_output, training=training)
#             attn_output = self.layernorm1(x + attn_output)
#             # Feed Forward Network
#             ffn_output = self.dense1(attn_output)
#             ffn_output = self.dense2(ffn_output)
#             ffn_output = self.dropout2(ffn_output, training=training)
#             ffn_output = self.layernorm2(attn_output + ffn_output)
#             x = ffn_output
#         return x


# def SSViT(
#     WS,
#     k,
#     Num_Classes,
#     num_transformer_layers=4,
#     num_heads=8,
#     dff=4 * 64,
#     dropout_rate=0.1,
#     layernorm_eps=1e-3,
# ):
#     input_shape = (WS, WS, k)
#     max_patch_size = min(input_shape[:3])
#     patch_size = max_patch_size
#     num_patches = (
#         (input_shape[0] // patch_size)
#         * (input_shape[1] // patch_size)
#         * (input_shape[2] // patch_size)
#     )
#     embedding_dim = 64
#     inputs = Input(shape=input_shape, name="inputs")
#     x = tf.expand_dims(inputs, axis=1)  # add this line
#     # Create a patch embedding layer
#     patch_embed = Conv3D(
#         embedding_dim,
#         kernel_size=(1, patch_size, patch_size),
#         strides=(1, patch_size, patch_size),
#         padding="valid",
#         name="patch_embed",
#     )(x)
#     cls_token = tf.Variable(tf.zeros((1, 1, 1, embedding_dim)), name="cls_token")
#     # Add cls_token to patch embeddings
#     patch_embed = tf.concat(
#         [
#             tf.broadcast_to(
#                 cls_token, [tf.shape(patch_embed)[0], 1, 1, 1, embedding_dim]
#             ),
#             patch_embed,
#         ],
#         axis=1,
#     )
#     # Add positional encoding to patch embeddings
#     pos_encoding = get_positional_encoding_ssvit(num_patches + 1, embedding_dim)
#     patch_embed = patch_embed + pos_encoding
#     # Create a Transformer_SSViT encoder
#     for i in range(num_transformer_layers):
#         if i == 0:
#             transformer_layer = Transformer_SSViT(
#                 num_heads=num_heads,
#                 transformer_layer_depth=1,
#                 d_model=embedding_dim,
#                 dff=dff,
#                 name=f"transformer_layer_{i}",
#             )(patch_embed)
#         else:
#             transformer_layer = Transformer_SSViT(
#                 num_heads=num_heads,
#                 transformer_layer_depth=1,
#                 d_model=embedding_dim,
#                 dff=dff,
#                 name=f"transformer_layer_{i}",
#             )(transformer_layer)

#     # Use cross-attention between the patch embeddings and a separate set of learnable embeddings
#     q = Dense(embedding_dim)(transformer_layer[:, 0, 0, 0])
#     k = Dense(embedding_dim)(patch_embed[:, 1:, :, :, :])
#     v = patch_embed[:, 1:, :, :, :]
#     cross_attn_output = tf.matmul(q[:, tf.newaxis, :], k, transpose_b=True)
#     cross_attn_output = cross_attn_output / tf.math.sqrt(
#         tf.cast(embedding_dim, dtype=tf.float32)
#     )
#     cross_attn_output = tf.nn.softmax(cross_attn_output, axis=-1)
#     cross_attn_output = tf.matmul(cross_attn_output, v)
#     # Concatenate cls_token and cross-attention output
#     output_layer = tf.concat(
#         [transformer_layer[:, 0, 0, 0], cross_attn_output[:, 0, 0, 0]], axis=-1
#     )
#     # Add a few dense layers and a final softmax layer
#     output_layer = Dense(dff, activation="relu")(output_layer)
#     output_layer = Dense(dff, activation="relu")(output_layer)
#     output_layer = Dense(Num_Classes, activation="softmax", name="output_layer")(
#         output_layer
#     )
#     # Define the model
#     model = Model(inputs=inputs, outputs=output_layer, name="SSViT")
#     return model
# # SSViT Ends
# # Pyformer Starts
# def pyramid_3d_cnn(WS, k, num_pyramid_levels):
#     input_shape = (WS, WS, k, 1)
#     max_patch_size = min(input_shape[:3])
#     patch_size = max_patch_size
#     inputs = layers.Input(shape=input_shape)
#     x = inputs
#     pyramid_outputs = []
#     for level in range(num_pyramid_levels):
#         scale = 2 ** level
#         scaled_input_shape = (input_shape[0] // scale, input_shape[1] // scale, input_shape[2] // scale, input_shape[3])
#         x = tf.keras.layers.Conv3D(32, kernel_size=(1, patch_size, patch_size), activation='relu', padding='same', input_shape=scaled_input_shape)(x)
#         x = tf.keras.layers.Conv3D(64, kernel_size=(1, patch_size, patch_size), activation='relu', padding='same')(x)
#         pyramid_outputs.append(x)
#     merged = tf.keras.layers.concatenate(pyramid_outputs, axis=-1)
#     return Model(inputs=inputs, outputs=merged)

# def PyFormerModel(WS, k, Num_Classes):
#     num_pyramid_levels=2
#     num_layers = 2
#     num_heads = 4
#     mlp_dim = 64
#     input_shape = (WS, WS, k, 1)
#     inputs = layers.Input(shape=input_shape)
#     x = pyramid_3d_cnn(WS, k, num_pyramid_levels)(inputs)
#     x = layers.Activation("relu")(x)
#     for i in range(num_layers):
#         x = layers.LayerNormalization()(x)
#         x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=k)(x, x)
#         x = layers.Add()([x, inputs])
#         y = layers.Conv3D(filters=k, kernel_size=(1, WS, WS), activation='relu', padding='same')(x)
#         y = layers.Conv3D(filters=2*mlp_dim, kernel_size=(1, WS, WS), activation='relu', padding='same')(y)
#         x = layers.Add()([x, y])
#     flatten_layer = layers.Flatten()(x)
#     x = layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(flatten_layer)
#     outputs = layers.Dense(Num_Classes, activation="softmax")(x)
#     model = Model(inputs=inputs, outputs=outputs, name='PyFormer')
#     return model
# # Pyformer Model Ends

# # Attention GCN Starts
# def AttentionGCN(WS, k, Num_Classes):

#     input_layer = Input(shape=(WS,WS,k), name='input')
#     #inputs = data_augmentation(input_layer)
#     conv1 = Conv2D(8, kernel_size=1, activation='relu', name='conv1', kernel_regularizer=regularizers.l2(0.0005))(input_layer)
#     nn = attention_layers.cot_attention(conv1, kernel_size=3)
#     #pool1 = MaxPooling2D(pool_size=(2,2), name='pool1')(nn)
# ########## Graph-based global reasoning module # input is 8x8x8
#     graph_t = Reshape((8, WS*WS))(conv1)
#     squeezed_graph_t = Conv1D(16, 1, activation='relu', name='squeezer', kernel_regularizer=regularizers.l2(0.0001))(graph_t)
#     squeezed_graph = Reshape((8, 16))(squeezed_graph_t)
#     gconv = Conv1D(64, 1, activation='relu', name='gconv', kernel_regularizer=regularizers.l2(0.0001))(squeezed_graph)
#     gconv_t = Reshape((8, 64))(gconv)
#     unsqueezed_graph_transpose = Conv1D(WS*WS, 1, activation='relu', name='unsqueezer', kernel_regularizer=regularizers.l2(0.0001))(gconv_t)
#     unsqueezed_graph = Reshape((WS*WS, 8))(unsqueezed_graph_transpose)
#     glore = Reshape((WS, WS, 8))(unsqueezed_graph)

#     block1 = layers.concatenate([conv1, glore], axis=-1) #################  you do not need to add the feature maps of the conv1 layer
#     block12 = layers.concatenate([conv1, block1, nn], axis=-1) #################  you do not need to use the feature maps of the attention block
#     #pool2 = MaxPooling2D(pool_size=(2,2), name='pool2')(block1)
#     conv2 = Conv2D(32, kernel_size=1, activation='relu', name='conv2', kernel_regularizer=regularizers.l2(0.0005))(block12)

# # Graph-based global reasoning module 2 # input is 8x8x32

#     graph_t2 = Reshape((32,WS*WS))(conv2)
#     squeezed_graph_t2 = Conv1D(64, 1, activation='relu', name='squeezer2', kernel_regularizer=regularizers.l2(0.0001))(graph_t2)
#     squeezed_graph2 = Reshape((64, 32))(squeezed_graph_t2)
#     gconv2 = Conv1D(32, 1, activation='relu', name='gconv2', kernel_regularizer=regularizers.l2(0.0001))(squeezed_graph2)
#     gconv_transpose2 = Reshape((64, 32))(gconv2)
#     unsqueezed_graph_t2 = Conv1D(WS*WS, 1, activation='relu', name='unsqueezer2', kernel_regularizer=regularizers.l2(0.0001))(gconv_transpose2)
#     unsqueezed_graph2 = Reshape((WS*WS, 64))(unsqueezed_graph_t2)
#     glore2 = Reshape((WS, WS, 64))(unsqueezed_graph2)

#     block2 = layers.concatenate([conv2, glore2], axis=-1) #################  you do not need to add the feature maps of the conv2 layer
#     block21 = layers.concatenate([conv2, block2, nn], axis=-1) #################  you do not need to use the feature maps of the attention block2

#     conv3 = Conv2D(32, kernel_size=1, activation='relu', name='conv3', kernel_regularizer=regularizers.l2(0.0005))(block21)
#     conv4 = Conv2D(64, kernel_size=3, activation='relu', name='conv4', kernel_regularizer=regularizers.l2(0.0005))(conv3)
#     conv5 = Conv2D(128, kernel_size=1, activation='relu', name='conv5', kernel_regularizer=regularizers.l2(0.0005))(conv4)

#     flat = Flatten()(conv5)
#     fc1 = Dense(100, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01))(flat)
#     fc2 = Dense(20, activation='relu', name='fc2', kernel_regularizer=regularizers.l2(0.01))(fc1)
#     fc3 = Dense(Num_Classes, activation='softmax', name='fc3', kernel_regularizer=regularizers.l2(0.0001))(fc2)

#     model = Model(inputs=input_layer, outputs=fc3, name='CNN_model_with_global_reasoning_module')
#     return model
# # Attention GCN Ends

# #Diff- Former Starts
# class SWiGLU(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(SWiGLU, self).__init__(**kwargs)

#     def call(self, inputs):
#         x, gate = tf.split(inputs, 2, axis=-1)
#         return x * tf.sigmoid(gate) + x

# def Positional_Encoding(max_len, d_emb):
#     pos = tf.cast(tf.range(max_len)[:, tf.newaxis], tf.float32)
#     i = tf.cast(tf.range(d_emb)[tf.newaxis, :], tf.float32)
#     angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_emb, tf.float32))
#     angles = pos * angle_rates
#     sines = tf.sin(angles[:, 0::2])
#     cosines = tf.cos(angles[:, 1::2])
#     pos_encoding = tf.concat([sines, cosines], axis=-1)
#     return pos_encoding[tf.newaxis, ...]

# class DMHSA(tf.keras.layers.Layer):
#     def __init__(self, num_heads, d_model, dropout_rate=0.1, **kwargs):
#         super(DMHSA, self).__init__(**kwargs)
#         self.num_heads = num_heads
#         self.d_model = d_model
#         self.depth = d_model // num_heads
#         self.wq = Dense(d_model)
#         self.wk = Dense(d_model)
#         self.wv = Dense(d_model)
#         self.dense = Dense(d_model)
#         self.dropout = Dropout(dropout_rate)
#     def split_heads(self, x, batch_size):
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
#         return tf.transpose(x, perm=[0, 2, 1, 3])
#     def differential_attention(self, scores):
#         diff_scores = scores[:, 1:] - scores[:, :-1]
#         diff_scores = tf.pad(diff_scores, [[0, 0], [1, 0], [0, 0], [0, 0]])
#         return diff_scores
#     def call(self, q, k, v, training):
#         batch_size = tf.shape(q)[0]
#         q = self.split_heads(self.wq(q), batch_size)
#         k = self.split_heads(self.wk(k), batch_size)
#         v = self.split_heads(self.wv(v), batch_size)
#         scores = tf.matmul(q, k, transpose_b=True)
#         scores /= tf.math.sqrt(tf.cast(self.depth, tf.float32))
#         diff_scores = self.differential_attention(scores)
#         scores = scores + diff_scores
#         attention_weights = tf.nn.softmax(scores, axis=-1)
#         attention_weights = self.dropout(attention_weights, training=training)
#         output = tf.matmul(attention_weights, v)
#         output = tf.transpose(output, perm=[0, 2, 1, 3])
#         output = tf.reshape(output, (batch_size, -1, self.d_model))
#         return self.dense(output)

# class SST(tf.keras.layers.Layer):
#     def __init__(self, num_heads, transformer_layer_depth, d_model, dff, dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
#         super(SST, self).__init__(**kwargs)
#         self.num_heads = num_heads
#         self.transformer_layer_depth = transformer_layer_depth
#         self.d_model = d_model
#         self.dff = dff
#         self.dropout_rate = dropout_rate
#         self.layernorm_eps = layernorm_eps
#         self.mha = DMHSA(num_heads=num_heads, d_model=d_model, dropout_rate=dropout_rate)
#         self.dropout1 = Dropout(dropout_rate)
#         self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
#         self.dense1 = Dense(dff)
#         self.swiGLU1 = SWiGLU()
#         self.dense2 = Dense(d_model)
#         self.dropout2 = Dropout(dropout_rate)
#         self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
#     def call(self, inputs, training):
#         x = inputs
#         for _ in range(self.transformer_layer_depth):
#             attn_output = self.mha(x, x, x, training=training)
#             attn_output = self.dropout1(attn_output, training=training)
#             attn_output = self.layernorm1(x + attn_output)
#             ffn_output = self.dense1(attn_output)
#             ffn_output = self.swiGLU1(ffn_output)
#             ffn_output = self.dense2(ffn_output)
#             ffn_output = self.dropout2(ffn_output, training=training)
#             ffn_output = self.layernorm2(attn_output + ffn_output)
#             x = ffn_output
#         return x

# def DSST(WS, k, Num_Classes, num_transformer_layers=2, num_heads=num_heads, dff=4 * 64, dropout_rate=0.1, layernorm_eps=1e-3):
#     input_shape = (WS, WS, k)
#     max_patch_size = min(input_shape[:3])
#     patch_size = max_patch_size
#     num_patches = ((input_shape[0] // patch_size) * (input_shape[1] // patch_size) * (input_shape[2] // patch_size))
#     embedding_dim = 64
#     inputs = Input(shape=input_shape, name="inputs")
#     x = tf.expand_dims(inputs, axis=1)
#     patch_embed = Conv3D(embedding_dim, kernel_size=(1, patch_size, patch_size),
#                          strides=(1, patch_size, patch_size), padding="valid", name="patch_embed")(x)
#     patch_embed = tf.reshape(patch_embed, (-1, tf.reduce_prod(tf.shape(patch_embed)[1:-1]), embedding_dim)) 
#     cls_token = tf.Variable(tf.zeros((1, 1, embedding_dim)), name="cls_token")
#     patch_embed = tf.concat([tf.broadcast_to(cls_token, [tf.shape(patch_embed)[0], 1, embedding_dim]), patch_embed], axis=1)
#     # pos_encoding = Positional_Encoding(num_patches + 1, embedding_dim)
#     seq_len = tf.shape(patch_embed)[1]
#     pos_encoding = Positional_Encoding(seq_len, embedding_dim)
#     pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
#     pos_encoding = tf.reshape(pos_encoding, (1, -1, embedding_dim))
#     patch_embed = patch_embed + pos_encoding
#     transformer_layer = patch_embed
#     for i in range(num_transformer_layers):
#         transformer_layer = SST(
#             num_heads=num_heads,
#             transformer_layer_depth=2,
#             d_model=embedding_dim,
#             dff=dff,
#             name=f"transformer_layer_{i}",
#         )(transformer_layer)

#     transformer_layer = tf.reshape(transformer_layer, (-1, tf.shape(transformer_layer)[1], embedding_dim))
#     cls_token_output = transformer_layer[:, 0, :] 
#     output_layer = Dense(32, kernel_regularizer=regularizers.l2(0.01))(cls_token_output)
#     output_layer = Dense(Num_Classes, activation="softmax", name="output_layer")(output_layer)
#     model = Model(inputs=inputs, outputs=output_layer, name="DSST")
#     return model
# # Diff-Former Ends

# # SSFKANMAMBA Starts
# # # Token Generation
# class SpectralSpatialTokenGeneration(tf.keras.layers.Layer):
#     '''
#     token generation for spectral and spatial features
#     fully connected layer for spectral and spatial features
#     '''
#     def __init__(self, out_channels, **kwargs):
#         super(SpectralSpatialTokenGeneration, self).__init__(**kwargs)
#         self.spatial_tokens = Dense(out_channels)   # pytorch nn.Linear
#         self.spectral_tokens = Dense(out_channels)
#     def call(self, x):
#         B, H, W, C = x.shape
#         # Use tf.shape to handle dynamic batch dimension
#         spatial_tokens = self.spatial_tokens(tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [tf.shape(x)[0], H * W, C]))
#         spectral_tokens = self.spectral_tokens(tf.reshape(tf.transpose(x, [0, 1, 2, 3]), [tf.shape(x)[0], H * W, C]))
#         return spatial_tokens, spectral_tokens

# # Wave Mamba 
# class WaveSpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
#     def __init__(self, out_channels, **kwargs):  # Add 'self'
#         super(WaveSpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
#         self.spatial_gate = Sequential([
#             Dense(out_channels),
#             Activation('sigmoid'),
#             # Reshape the output to match spatial_tokens
#             Reshape((1, out_channels))
#         ])
#         self.spectral_gate = Sequential([
#             Dense(out_channels),
#             Activation('sigmoid'),
#             # Reshape the output to match spectral_tokens
#             Reshape((1, out_channels))
#         ])
#     def call(self, spatial_tokens, spectral_tokens, center_tokens):
#         spatial_enhanced = spatial_tokens * self.spatial_gate(center_tokens)
#         spectral_enhanced = spectral_tokens * self.spectral_gate(center_tokens)
#         return spatial_enhanced, spectral_enhanced

# class WaveStateSpaceModel(tf.keras.layers.Layer):
#     def __init__(self, state_dim, **kwargs):
#         super(WaveStateSpaceModel, self).__init__(**kwargs)
#         self.state_dim = state_dim
#         self.state_transition = Dense(units=state_dim, activation="relu")
#         self.state_update = Dense(units=state_dim, activation="relu")
#     def call(self, x):
#         state = tf.zeros([tf.shape(x)[0], self.state_dim])
#         for t in range(tf.shape(x)[1]):
#             # Flatten the input if it has more than 2 dimensions
#             input_t = tf.reshape(x[:, t, :], [tf.shape(x)[0], -1])
#             state = self.state_transition(state) + self.state_update(input_t)
#         return state

# class WaveSSMambaModel(tf.keras.Model):
#     def __init__(self, out_channels, num_heads, state_dim, Num_Classes, dropout=0.1, **kwargs):
#         super(WaveSSMambaModel, self).__init__(**kwargs)
#         self.token_generation = SpectralSpatialTokenGeneration(out_channels)
#         self.feature_enhancement = WaveSpectralSpatialFeatureEnhancement(out_channels)
#         self.state_space_model = WaveStateSpaceModel(state_dim)
#         self.dense = Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
#         self.dropout = Dropout(0.4)
#         self.classifier = Dense(Num_Classes, activation='softmax')
#     def apply_wavelet_transform(self, image, wavelet_name='haar'):
#         """'haar', 'db1', 'db2', 'coif1', 'bior1.3'"""
#         image_np = tf.make_ndarray(tf.make_tensor_proto(image))
#         if wavelet_name not in pywt.wavelist(kind='discrete'):
#             raise ValueError(f"The wavelet {wavelet_name} is not supported for discrete wavelet transform.")
#         coeffs = pywt.dwt2(image_np, wavelet_name)
#         cA, (cH, cV, cD) = coeffs
#         transformed = np.stack([cA, cH, cV, cD], axis=-1)
#         return tf.convert_to_tensor(transformed, dtype=tf.float32)
#     def call(self, x):
#         spatial_tokens, spectral_tokens = self.token_generation(x)
#         center_tokens = spatial_tokens[:, x.shape[1] // 2, :]
#         spatial_tokens, spectral_tokens = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
#         spatial_tokens = tf.map_fn(lambda t: tf.numpy_function(self.apply_wavelet_transform, [t], tf.float32), spatial_tokens, dtype=tf.float32)
#         spectral_tokens = tf.map_fn(lambda t: tf.numpy_function(self.apply_wavelet_transform, [t], tf.float32), spectral_tokens, dtype=tf.float32)
#         Combined_Tokens = tf.concat([spatial_tokens, spectral_tokens], axis=-1)
#         state_output = self.state_space_model(Combined_Tokens)
#         output = self.classifier(self.dense(state_output))
#         return output

# def WaveMamba(Tr, batch_size, Num_Classes):
#     model = WaveSSMambaModel(out_channels=32, num_heads=num_heads, state_dim=64, Num_Classes=Num_Classes, dropout=0.1)
#     sample_input = Tr[:batch_size]
#     _ = model(sample_input)
#     return model
# # Wave Mamba Ends

# # Multihead Mamba Starts
# class MulMultiHeadAttention(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs): # Add 'self'
#         super(MulMultiHeadAttention, self).__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.all_head_size = self.num_heads * self.head_dim

#         self.query = Dense(self.all_head_size)
#         self.key = Dense(self.all_head_size)
#         self.value = Dense(self.all_head_size)
#         self.dropout = Dropout(dropout)

#     def call(self, query, key, value):
#         batch_size = tf.shape(query)[0]

#         query_proj = self.query(query)
#         key_proj = self.key(key)
#         value_proj = self.value(value)

#         query_proj = tf.reshape(query_proj, [batch_size, -1, self.num_heads, self.head_dim])
#         query_proj = tf.transpose(query_proj, [0, 2, 1, 3])
#         key_proj = tf.reshape(key_proj, [batch_size, -1, self.num_heads, self.head_dim])
#         key_proj = tf.transpose(key_proj, [0, 2, 1, 3])
#         value_proj = tf.reshape(value_proj, [batch_size, -1, self.num_heads, self.head_dim])
#         value_proj = tf.transpose(value_proj, [0, 2, 1, 3])

#         attention_scores = tf.matmul(query_proj, key_proj, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
#         attention_weights = tf.nn.softmax(attention_scores, axis=-1)
#         attention_output = tf.matmul(attention_weights, value_proj)

#         attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
#         attention_output = tf.reshape(attention_output, [batch_size, -1, self.all_head_size])
#         attention_output = self.dropout(attention_output)

#         return attention_output

# class MulSpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
#     def __init__(self, out_channels, **kwargs):  # Add 'self'
#         super(MulSpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
#         self.spatial_gate = Sequential([
#             Dense(out_channels),
#             Activation('sigmoid'),
#             # Reshape the output to match spatial_tokens
#             Reshape((1, out_channels))
#         ])
#         self.spectral_gate = Sequential([
#             Dense(out_channels),
#             Activation('sigmoid'),
#             # Reshape the output to match spectral_tokens
#             Reshape((1, out_channels))
#         ])

#     def call(self, spatial_tokens, spectral_tokens, center_tokens):
#         spatial_enhanced = spatial_tokens * self.spatial_gate(center_tokens)
#         spectral_enhanced = spectral_tokens * self.spectral_gate(center_tokens)
#         return spatial_enhanced, spectral_enhanced

# class MulStateSpaceModel(tf.keras.layers.Layer):
#     def __init__(self, state_dim, **kwargs):  # Add 'self'
#         super(MulStateSpaceModel, self).__init__(**kwargs)
#         self.state_dim = state_dim
#         self.state_transition = Dense(state_dim)
#         self.state_update = Dense(state_dim)

#     def call(self, x):
#         state = tf.zeros([tf.shape(x)[0], self.state_dim])
#         for t in range(tf.shape(x)[1]):
#             state = self.state_transition(state) + self.state_update(x[:, t, :])
#         return state

# class MulSSMambaModel(tf.keras.Model):
#     def __init__(self, out_channels, num_heads, state_dim, dropout=0.1, **kwargs): # Pass out_channels to __init__
#         super(MulSSMambaModel, self).__init__(**kwargs)
#         self.token_generation = SpectralSpatialTokenGeneration(out_channels) # Use out_channels here
#         self.multi_head_attention = MulMultiHeadAttention(out_channels, num_heads, dropout)
#         self.feature_enhancement = MulSpectralSpatialFeatureEnhancement(out_channels)
#         self.state_space_model = MulStateSpaceModel(state_dim)
#         self.dense = Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
#         self.dropout = Dropout(0.4)
#         self.classifier = Dense(Num_Classes, activation='softmax')

#     def call(self, x):
#         spatial_tokens, spectral_tokens = self.token_generation(x)
#         center_tokens = spatial_tokens[:, x.shape[1] // 2, :]
#         spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
#         attention_output = self.multi_head_attention(spatial_enhanced, spectral_enhanced, spectral_enhanced)
#         state_output = self.state_space_model(attention_output)
#         output = self.classifier(self.dense(state_output))
#         return output

# def MulMamba(Tr, batch_size):
#     model = MulSSMambaModel(out_channels=32, num_heads=num_heads, state_dim=64, dropout=0.1)
#     sample_input = Tr[:batch_size]
#     _ = model(sample_input)
#     return model
# # Multihead Mamba Ends


# # SSMamba Starts
# # Spaital Spectral Mamba
# class SSMSpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
#     def __init__(self, out_channels, **kwargs):
#         super(SSMSpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
#         self.spatial_gate = Sequential([
#             Dense(out_channels),
#             Activation('sigmoid'),
#             Reshape((1, out_channels))
#         ])
#         self.spectral_gate = Sequential([
#             Dense(out_channels),
#             Activation('sigmoid'),
#             Reshape((1, out_channels))
#         ])
#     def call(self, spatial_tokens, spectral_tokens, center_tokens):
#         spatial_enhanced = spatial_tokens * self.spatial_gate(center_tokens)
#         spectral_enhanced = spectral_tokens * self.spectral_gate(center_tokens)
#         return spatial_enhanced, spectral_enhanced

# class SSMStateSpaceModel(tf.keras.layers.Layer):
#     def __init__(self, state_dim, **kwargs):
#         super(SSMStateSpaceModel, self).__init__(**kwargs)
#         self.state_dim = state_dim
#         self.state_transition = Dense(state_dim)
#         self.state_update = Dense(state_dim)
#     def call(self, x):
#         state = tf.zeros([tf.shape(x)[0], self.state_dim])
#         for t in range(tf.shape(x)[1]):
#             state = self.state_transition(state) + self.state_update(x[:, t, :])
#         return state

# class SSMambaModel(tf.keras.Model):
#     def __init__(self, out_channels,state_dim, dropout=0.1, **kwargs):
#         super(SSMambaModel, self).__init__(**kwargs)
#         self.token_generation = SpectralSpatialTokenGeneration(out_channels)
#         # self.multi_head_attention = MultiHeadAttention(out_channels, num_heads, dropout)
#         self.feature_enhancement = SSMSpectralSpatialFeatureEnhancement(out_channels)
#         self.state_space_model = SSMStateSpaceModel(state_dim)
#         self.dense = Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
#         self.dropout = Dropout(0.4)
#         self.classifier = Dense(Num_Classes, activation='softmax')
#     def call(self, x):
#         spatial_tokens, spectral_tokens = self.token_generation(x)
#         center_tokens = spatial_tokens[:, x.shape[1] // 2, :]
#         spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
#         # attention_output = self.multi_head_attention(spatial_enhanced, spectral_enhanced, spectral_enhanced)
#         # fused_tokens = tf.concat([spatial_enhanced, spectral_enhanced], axis=-1)
#         fuesd_tokens = spatial_enhanced + spectral_enhanced
#         state_output = self.state_space_model(fuesd_tokens)
#         output = self.classifier(state_output)
#         return output
    
# def SSM(Tr, batch_size):
#     model = SSMambaModel(
#         out_channels=32,
#         state_dim=64,
#         dropout=0.1
#     )
#     # Build the model by passing a batch of data
#     _ = model(Tr[:batch_size])  # Ensures model is built correctly
#     return model

#  # Flops Calculation
# def get_flops(model, input_shape=None):
#     total_flops = 0

#     # Build model with dummy input to infer shapes if input_shape is provided
#     if input_shape and isinstance(model, tf.keras.Model) and not model.built:
#         dummy_input = tf.keras.Input(shape=input_shape)
#         model(dummy_input)

    

#     def process_layer(layer, input_shape=None):
#         nonlocal total_flops
#         flops = 0

#         # Helper to get input/output shapes
#         def get_shapes(layer):
#             in_shape = layer.input_shape if hasattr(layer, 'input_shape') else input_shape
#             out_shape = layer.output_shape if hasattr(layer, 'output_shape') else None
#             return in_shape, out_shape

#         # Handle Conv2D layers
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             in_shape, out_shape = get_shapes(layer)
#             if in_shape and out_shape:
#                 input_channels = in_shape[-1]
#                 output_height = out_shape[1]
#                 output_width = out_shape[2]
#                 kernel_size = layer.kernel_size[0] * layer.kernel_size[1] * input_channels
#                 flops += kernel_size * output_height * output_width * layer.filters * 2
#                 if layer.activation is not None and layer.activation.__name__ != 'linear':
#                     flops += output_height * output_width * layer.filters

#         # Handle Conv3D layers
#         elif isinstance(layer, tf.keras.layers.Conv3D):
#             in_shape, out_shape = get_shapes(layer)
#             if in_shape and out_shape:
#                 input_channels = in_shape[-1]
#                 output_depth = out_shape[1]
#                 output_height = out_shape[2]
#                 output_width = out_shape[3]
#                 kernel_size = layer.kernel_size[0] * layer.kernel_size[1] * layer.kernel_size[2] * input_channels
#                 flops += kernel_size * output_depth * output_height * output_width * layer.filters * 2
#                 if layer.activation is not None and layer.activation.__name__ != 'linear':
#                     flops += output_depth * output_height * output_width * layer.filters

#         # Handle Dense layers
#         elif isinstance(layer, tf.keras.layers.Dense):
#             in_shape, out_shape = get_shapes(layer)
#             input_dim = layer.get_weights()[0].shape[0] if layer.get_weights() else in_shape[-1] if in_shape else 0
#             output_dim = layer.units
#             flops += 2 * input_dim * output_dim  # Multiplications + additions
#             if layer.activation is not None and layer.activation.__name__ != 'linear':
#                 flops += output_dim  # Approximate activation cost

#         # AttenctionGCN Flops
#          # Handle Reshape layers
#         elif isinstance(layer, tf.keras.layers.Reshape):
#             flops += 0  # Reshape is just memory rearrangement, negligible FLOPs
#         # Handle Flatten layers
#         elif isinstance(layer, tf.keras.layers.Flatten):
#             flops += 0  # Flatten is just memory rearrangement, negligible FLOPs
#         # Handle concatenate layers
#         elif isinstance(layer, tf.keras.layers.Concatenate):
#             flops += 0  # Concatenate is just memory copying, negligible FLOPs
#         # Handle cot_attention (custom attention layer)
#         elif layer.__class__.__name__ == 'cot_attention':  # Assuming cot_attention is a custom layer
#             in_shape, out_shape = get_shapes(layer)
#             if in_shape and out_shape:
#                 height, width, channels = in_shape[1], in_shape[2], in_shape[3]
#                 # Assume cot_attention uses a 3x3 convolution to compute attention scores
#                 kernel_size = 3 * 3 * channels  # 3x3 kernel
#                 attention_flops = kernel_size * height * width * channels * 2  # Conv for attention scores
#                 # Softmax approximation
#                 attention_flops += height * width * channels  # One operation per element
#                 # Weighted sum (attention weights * input)
#                 attention_flops += 2 * height * width * channels * height * width  # Matrix multiplication
#                 flops += attention_flops
#         # Handle LayerNormalization
#         elif isinstance(layer, tf.keras.layers.LayerNormalization):
#             in_shape, _ = get_shapes(layer)
#             if in_shape:
#                 flops += 5 * np.prod(in_shape[1:])  # Mean, variance, scale, shift, normalize
#         # Handle Dropout (negligible FLOPs)
#         elif isinstance(layer, tf.keras.layers.Dropout):
#             flops += 0
#         # Handle functional model operations
#         elif isinstance(layer, tf.keras.Model):
#             for sublayer in layer.layers:
#                 if sublayer.name.startswith('concatenate'):
#                     flops += 0  # Negligible
#                 else:
#                     flops += process_layer(sublayer, input_shape)

#         # Handle MultiHeadAttention layers
#         elif isinstance(layer, tf.keras.layers.MultiHeadAttention):
#             in_shape, _ = get_shapes(layer)
#             embed_dim = layer._key_dim * layer._num_heads  # Use _num_heads
#             seq_len = in_shape[-2] if in_shape and len(in_shape) > 2 else 0
#             if seq_len:
#                 # Q, K, V projections
#                 flops += 3 * (2 * embed_dim * embed_dim * seq_len)  # Three dense layers
#                 # Scaled dot-product attention
#                 flops += 2 * layer._num_heads * seq_len * seq_len * (embed_dim // layer._num_heads)  # QK^T matmul
#                 flops += layer._num_heads * seq_len * seq_len  # Softmax (approx)
#                 flops += 2 * layer._num_heads * seq_len * seq_len * (embed_dim // layer._num_heads)  # Attention * V
#                 # Output projection
#                 flops += 2 * embed_dim * embed_dim * seq_len

#         # Handle Transformer/Transformer_SSViT layers
#         elif isinstance(layer, (Transformer, Transformer_SSViT)):
#             for _ in range(layer.transformer_layer_depth):
#                 flops += process_layer(layer.mha, input_shape)
#                 flops += process_layer(layer.dropout1, input_shape)
#                 flops += process_layer(layer.layernorm1, input_shape)
#                 flops += process_layer(layer.dense1, input_shape)
#                 flops += process_layer(layer.dense2, input_shape)
#                 flops += process_layer(layer.dropout2, input_shape)
#                 flops += process_layer(layer.layernorm2, input_shape)

#         # Handle LayerNormalization
#         elif isinstance(layer, tf.keras.layers.LayerNormalization):
#             in_shape, _ = get_shapes(layer)
#             if in_shape:
#                 flops += 5 * np.prod(in_shape[1:])  # Mean, variance, scale, shift, normalize

#         # Handle Dropout (negligible FLOPs)
#         elif isinstance(layer, tf.keras.layers.Dropout):
#             flops += 0

#         # Handle MulMultiHeadAttention
#         elif isinstance(layer, MulMultiHeadAttention):
#             flops += process_layer(layer.query)  # Dense layer
#             flops += process_layer(layer.key)    # Dense layer
#             flops += process_layer(layer.value)  # Dense layer
#             flops += process_layer(layer.dropout)  # Dropout layer (typically 0 FLOPs)
#             # Add FLOPs for attention computation
#             batch_size = 1  # Assume batch_size=1 for simplicity; adjust if known
#             seq_len = layer.embed_dim // layer.num_heads  # Approximation
#             flops += (2 * batch_size * layer.num_heads * seq_len * seq_len)  # QK^T matmul
#             flops += (batch_size * layer.num_heads * seq_len * seq_len)      # Softmax (approx.)
#             flops += (2 * batch_size * layer.num_heads * seq_len * seq_len)
            
#         # Handle MulSpectralSpatialFeatureEnhancement
#         elif isinstance(layer, MulSpectralSpatialFeatureEnhancement):
#             flops += process_layer(layer.spatial_gate, input_shape)
#             flops += process_layer(layer.spectral_gate, input_shape)
#             in_shape, _ = get_shapes(layer)
#             if in_shape:
#                 seq_len = in_shape[1] if len(in_shape) > 1 else 1
#                 out_channels = layer.spatial_gate.layers[0].units
#                 flops += 2 * in_shape[0] * seq_len * out_channels  # spatial_tokens * gate, spectral_tokens * gate
#          # Handle MulStateSpaceModel
#         elif isinstance(layer, MulStateSpaceModel):
#             flops += process_layer(layer.state_transition, input_shape)
#             flops += process_layer(layer.state_update, input_shape)
#             in_shape, _ = get_shapes(layer)
#             if in_shape:
#                 seq_len = in_shape[1] if len(in_shape) > 1 else 1
#                 flops += seq_len * (flops + in_shape[0] * layer.state_dim)  # State updates per timestep

#         # Handle WaveSpectralSpatialFeatureEnhancement
#         elif isinstance(layer, WaveSpectralSpatialFeatureEnhancement):
#             flops += process_layer(layer.spatial_gate, input_shape)
#             flops += process_layer(layer.spectral_gate, input_shape)
#             in_shape, _ = get_shapes(layer)
#             if in_shape:
#                 seq_len = in_shape[1] if len(in_shape) > 1 else 1
#                 out_channels = layer.spatial_gate.layers[0].units
#                 flops += 2 * in_shape[0] * seq_len * out_channels  # spatial_tokens * gate, spectral_tokens * gate
#         # Handle WaveStateSpaceModel
#         elif isinstance(layer, WaveStateSpaceModel):
#             flops += process_layer(layer.state_transition, input_shape)
#             flops += process_layer(layer.state_update, input_shape)
#             in_shape, _ = get_shapes(layer)
#             if in_shape:
#                 seq_len = in_shape[1] if len(in_shape) > 1 else 1
#                 flops += seq_len * (flops + in_shape[0] * layer.state_dim)  # State updates per timestep

#         # Handle WaveSSMambaModel
#         elif isinstance(layer, WaveSSMambaModel):
#             flops += process_layer(layer.token_generation, input_shape)
#             flops += process_layer(layer.feature_enhancement, input_shape)
#             in_shape, _ = get_shapes(layer)
#             if in_shape:
#                 batch_size = in_shape[0]
#                 height = in_shape[1]
#                 width = in_shape[2]
#                 channels = in_shape[3] if len(in_shape) > 3 else 1
#                 wavelet_flops_per_pixel = 8
#                 flops_per_transform = wavelet_flops_per_pixel * height * width * channels
#                 num_tokens = height
#                 flops += 2 * batch_size * num_tokens * flops_per_transform  # 2x for spatial and spectral
#             flops += process_layer(layer.state_space_model, input_shape)
#             flops += process_layer(layer.dense, input_shape)
#             flops += process_layer(layer.dropout, input_shape)
#             flops += process_layer(layer.classifier, input_shape)

#         # Handle SSMambaModel
#         elif isinstance(layer, SSMambaModel):
#             flops += process_layer(layer.token_generation, input_shape)
#             flops += process_layer(layer.feature_enhancement, input_shape)
#             in_shape, _ = get_shapes(layer)
#             if in_shape:
#                 batch_size = in_shape[0]
#                 height = in_shape[1]
#                 width = in_shape[2]
#                 channels = in_shape[3] if len(in_shape) > 3 else 1
#                 wavelet_flops_per_pixel = 8
#                 flops_per_transform = wavelet_flops_per_pixel * height * width * channels
#                 num_tokens = height
#                 flops += 2 * batch_size * num_tokens * flops_per_transform  # 2x for spatial and spectral
#             flops += process_layer(layer.state_space_model, input_shape)
#             flops += process_layer(layer.dense, input_shape)
#             flops += process_layer(layer.dropout, input_shape)
#             flops += process_layer(layer.classifier, input_shape)
            
#         elif isinstance(layer, SSMSpectralSpatialFeatureEnhancement):
#             flops += process_layer(layer.spatial_gate, input_shape)
#             flops += process_layer(layer.spectral_gate, input_shape)
#             in_shape, _ = get_shapes(layer)
#             if in_shape:
#                 seq_len = in_shape[1] if len(in_shape) > 1 else 1
#                 out_channels = layer.spatial_gate.layers[0].units
#                 flops += 2 * in_shape[0] * seq_len * out_channels
#         # FKAN FLOPs Calculation Starts
#         # Handle SplineLinear
#         # elif isinstance(layer, SplineLinear):
#         #     flops += layer.in_features * layer.out_features * 2  # Multiplications + additions

#         # # Handle RadialBasisFunction
#         # elif isinstance(layer, RadialBasisFunction):
#         #     input_dim = layer.get_weights()[0].shape[0] if layer.get_weights() else 0
#         #     flops += 4 * input_dim * layer.num_grids  # Subtraction, squaring, division, exp

#         # # Handle FastKANLayer
#         # elif isinstance(layer, FastKANLayer):
#         #     input_dim = layer.input_dim
#         #     output_dim = layer.output_dim
#         #     num_grids = layer.rbf.num_grids
#         #     if layer.use_layernorm:
#         #         flops += 5 * input_dim  # LayerNorm approximation
#         #     rbf_flops = 4 * input_dim * num_grids
#         #     spline_flops = input_dim * num_grids * output_dim * 2
#         #     base_flops = 0
#         #     if layer.use_base_update:
#         #         base_flops += input_dim * output_dim * 2
#         #         base_flops += input_dim  # Activation
#         #     flops += rbf_flops + spline_flops + base_flops

#         # # Handle SpectralSpatialTokenGeneration
#         # elif isinstance(layer, SpectralSpatialTokenGeneration):
#         #     flops += process_layer(layer.spatial_tokens, input_shape)
#         #     flops += process_layer(layer.spectral_tokens, input_shape)

#         # Handle MultiHeadKANAttention
#         # elif isinstance(layer, MultiHeadKANAttention):
#         #     flops += process_layer(layer.query_kan)  # FastKANLayer
#         #     flops += process_layer(layer.key_kan)    # FastKANLayer
#         #     flops += process_layer(layer.value_kan)  # FastKANLayer
#         #     flops += process_layer(layer.output_projection)  # Dense layer
#         #     if layer.output_activation:
#         #         flops += layer.output_dim  # Activation cost
#         # # Handle SpectralSpatialFeatureEnhancement
#         # elif isinstance(layer, SpectralSpatialFeatureEnhancement):
#         #     flops += process_layer(layer.spatial_gate, input_shape)
#         #     flops += process_layer(layer.spectral_gate, input_shape)
#         #     in_shape, _ = get_shapes(layer)
#         #     if in_shape:
#         #         seq_len = in_shape[1] if len(in_shape) > 1 else 1
#         #         out_channels = layer.spatial_gate.layers[0].output_dim
#         #         flops += 2 * in_shape[0] * seq_len * out_channels  # spatial_tokens * gate, spectral_tokens * gate

#         # # Handle StateSpaceModel
#         # elif isinstance(layer, StateSpaceModel):
#         #     flops += process_layer(layer.state_transition, input_shape)
#         #     flops += process_layer(layer.state_update, input_shape)
#         #     in_shape, _ = get_shapes(layer)
#         #     if in_shape:
#         #         seq_len = in_shape[1] if len(in_shape) > 1 else 1
#         #         flops += seq_len * (flops + in_shape[0] * layer.state_dim)  # State updates per timestep
#         # # FKAN FLOPs Calculation Ends


#         # Handle functional model operations
#         elif isinstance(layer, tf.keras.Model):
#             for sublayer in layer.layers:
#                 if sublayer.name.startswith('add'):  # Positional encoding addition
#                     in_shape = sublayer.input_shape[0] if sublayer.input_shape else None
#                     if in_shape:
#                         flops += np.prod(in_shape[1:])  # Element-wise addition
#                 elif sublayer.name.startswith('matmul'):  # Cross-attention matmul
#                     in_shape_1 = sublayer.input_shape[0] if sublayer.input_shape else None
#                     in_shape_2 = sublayer.input_shape[1] if sublayer.input_shape and len(sublayer.input_shape) > 1 else None
#                     if in_shape_1 and in_shape_2:
#                         flops += 2 * in_shape_1[-1] * in_shape_2[-1] * in_shape_1[-2]  # Matmul
#                 elif sublayer.name.startswith('softmax'):  # Cross-attention softmax
#                     in_shape = sublayer.input_shape if sublayer.input_shape else None
#                     if in_shape:
#                         flops += np.prod(in_shape[1:])  # Approximate softmax
#                 elif sublayer.name.startswith('concatenate'):  # Concatenation
#                     flops += 0  # Negligible
#                 elif sublayer.name.startswith('expand_dims'):  # tf.expand_dims
#                     flops += 0  # Negligible
#                 else:
#                     flops += process_layer(sublayer, input_shape)

#         # Handle Sequential or nested models
#         elif isinstance(layer, tf.keras.Sequential) or hasattr(layer, 'layers'):
#             for sublayer in layer.layers:
#                 flops += process_layer(sublayer, input_shape)
      
#         total_flops += flops
#         print(f"Layer {layer.name}: {flops:,} FLOPs")
#         return flops

#     process_layer(model)
#     return total_flops

# def profile_model_flops(model, input_shape):
#     """Profile inference FLOPs for one sample using TensorFlow's graph profiler."""
#     input_signature = [
#         tf.TensorSpec(
#             shape=(1,) + tuple(input_shape),
#             dtype=tf.float32,
#             name="model_input",
#         )
#     ]

#     @tf.function(input_signature=input_signature)
#     def inference_graph(inputs):
#         return model(inputs, training=False)

#     concrete_function = inference_graph.get_concrete_function()
#     frozen_function = convert_variables_to_constants_v2(concrete_function)
#     options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#     options["output"] = "none"
#     profile = tf.compat.v1.profiler.profile(
#         graph=frozen_function.graph,
#         options=options,
#     )
#     return int(profile.total_float_ops) if profile is not None else 0


# def compute_flops(model):
#     return get_flops(model)   

# # Train and Evaluate Model
# def train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, output_dir, HSID, teRatio, k, WS, DLM, RDHSI, GT,Num_Classes,batch_size,epochs, target_names):
#     # Create the output directory if it doesn't exist
#     output_dir = os.path.join(output_dir, "") + model_name.__name__
#     os.makedirs(output_dir, exist_ok=True)
#     print(model_name.__name__)
#     input_shape = (WS, WS, k)
#     if model_name == PyFormerModel:
#         input_shape = (WS, WS, k, 1)
#         Tr = Tr.reshape(-1, WS, WS, k, 1)
#         Va = Va.reshape(-1, WS, WS, k, 1)
#         Te = Te.reshape(-1, WS, WS, k, 1)
#         CRDHSI = CRDHSI.reshape(-1, WS, WS, k, 1)
#         model = PyFormerModel(WS, k, Num_Classes)
#     # elif model_name == FastKanMamba:
#     #     Tr = Tr.reshape(-1, WS,WS,k)  # Flatten input for the model
#     #     Va = Va.reshape(-1, WS,WS,k)  # Flatten input for the model
#     #     Te = Te.reshape(-1, WS,WS,k) 
#     #     model = FastKanMamba(Tr, batch_size)
#     elif model_name == WaveMamba:
#         Tr = Tr.reshape(-1, WS, WS, k)
#         Va = Va.reshape(-1, WS, WS, k)
#         Te = Te.reshape(-1, WS, WS, k)
#         model = WaveMamba(Tr, batch_size, Num_Classes)
#     elif model_name == MulMamba:
#         model = MulMamba(Tr, batch_size)
#     elif model_name == SSM:
#         model = SSM(Tr, batch_size)
#     else:
#         model = model_name(WS,k,Num_Classes)
#     # Calling Custom Model
#     #model = model_name(WS, k, Num_Classes)
#     file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_model_summary_of_{model_name.__name__}.txt"
#     # Assuming `model` is your Keras model
#     total_params = int(model.count_params())
#     trainable_params = int(np.sum([
#         np.prod(weight.shape) for weight in model.trainable_weights
#     ]))
#     non_trainable_params = int(np.sum([
#         np.prod(weight.shape) for weight in model.non_trainable_weights
#     ]))
#     flops = profile_model_flops(model, input_shape=input_shape)
#     with open(os.path.join(output_dir, file_name),'w') as fh:
#         # Pass the file handle in as a lambda function to make it callable
#         model.summary(print_fn=lambda x: fh.write(x + '\n'))
#         fh.write(f"\nTotal Parameters: {total_params}\n")
#         fh.write(f"Trainable Parameters: {trainable_params}\n")
#         fh.write(f"Non-trainable Parameters: {non_trainable_params}\n")
#         fh.write(f"Total FLOPs (batch=1 inference): {flops}\n")
#         fh.write(f"GFLOPs (batch=1 inference): {flops / 1e9:.6f}\n")
        
#     # Compiling the model
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#     # Training the model
#     start = time.time()
#     history = model.fit(x=Tr, y=TrC, batch_size=batch_size, epochs=epochs, validation_data=(Va, VaC))
#     end = time.time()
#     Tr_Time = end - start
#     # Predicting with the model
#     start = time.time()
#     Te_Pred = model.predict(Te)
#     end = time.time()
#     Te_Time = end - start
#     ##Classification Report
#     classification,confusion,oa,each_acc,aa,kappa = ClassificationReports(TeC, Te_Pred, target_names,  zero_division=0)
#     print(classification)
#     #     # Writing Results in CSV File
#     file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_Classification_Report_{model_name.__name__}.csv"
#     CSV_file = CSVResults(os.path.join(output_dir, file_name), classification, confusion, Tr_Time, Te_Time, DL_Time, kappa, oa, aa,each_acc)
#     # #Confusion Matrix
#     cm, annot = Conf_Mat(Te_Pred, TeC, target_names)
#     cm.index.name = 'Actual'
#     cm.columns.name = 'Predicted'
#     fig, ax = plt.subplots(figsize=(15,15))
#     sns.heatmap(cm, cmap= "Spectral", annot=annot, fmt='', ax=ax, linewidths=0.5)
#     file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_Confusion_Matrix_{model_name.__name__}.png"
#     plt.savefig(os.path.join(output_dir, file_name), dpi=500)

#     # Preserve the existing t-SNE path for models that expose embeddings.
#     # PyFormer has no get_embeddings method, so only this optional plot is skipped.
#     if hasattr(model, "get_embeddings"):
#         embeddings = model.get_embeddings(Tr)
#         embeddings_np = embeddings.numpy()
#         tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
#         embeddings_2d = tsne.fit_transform(embeddings_np)
#         sns.set(style="whitegrid")
#         plt.figure(figsize=(8, 5))
#         plt.scatter(
#             embeddings_2d[:, 0],
#             embeddings_2d[:, 1],
#             c=TrC.argmax(axis=1),
#             cmap="viridis",
#         )
#         plt.title(f"tSNE: {HSID}", fontsize=12, fontweight="bold")
#         plt.xlabel("tSNE-1", fontsize=12, fontweight="bold")
#         plt.ylabel("tSNE-2", fontsize=12, fontweight="bold")
#         plt.tick_params(labelsize=12)
#         plt.grid(True)
#         plt.tight_layout()
#         file_name = f"{HSID}_{trRatio}_{k}_{WS}_tsne.png"
#         plt.savefig(
#             os.path.join(output_dir, file_name),
#             dpi=500,
#             format="png",
#             bbox_inches="tight",
#             pad_inches=0,
#         )
#         plt.close()

#     # Ground Truths
#     outputs = GT_Plot(CRDHSI, GT, model, WS, k)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(outputs, cmap='nipy_spectral')
#     plt.axis('off')
#     file_name = f"{HSID}_{trRatio}_{k}_{WS}_{DLM}_Ground_Truths.png"
#     plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)

#     return history, Tr_Time, Te_Time, oa, aa, kappa, total_params, flops


# # # Define a list of model names
# model_names = [PyFormerModel] 
# # Loop over the model names and run train_and_evaluate_model() for each model
# all_results = {}
# history_list = []
# Tr_Time_list= []
# Te_Time_list= []
# for model_name in model_names:
#     history, Tr_Time, Te_Time, oa, aa, kappa, total_params, flops = train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, output_dir, HSID, teRatio, k, WS, DLM, RDHSI, GT, Num_Classes, batch_size, epochs, target_names)
#     history_list.append(history)
#     Tr_Time_list.append(Tr_Time)
#     Tr_Time_list.append(Te_Time)

#     # Add results to the dictionary
#     model_function_name = model_name.__name__
#     all_results[model_function_name] = {
#         "history": history.history,
#         "Tr_Time": Tr_Time,
#         "Te_Time": Te_Time,
#         "Overall Accuracy": oa,
#         "Average Accuracy": aa,
#         "Kappa": kappa,
#         "Training Parameters": total_params,
#         "FLOPs (batch=1 inference)": flops,
#         "GFLOPs (batch=1 inference)": flops / 1e9,
#     }

# # Define the filename for all results
# all_results_filename = "all_models_results.json"

# # Concatenate output_dir with the filename
# all_results_output_path = os.path.join(output_dir, all_results_filename)

# # Convert the dictionary to JSON format
# json_data = json.dumps(all_results, indent=4)

# # Optionally, write the JSON data to a file
# with open(all_results_output_path, 'w') as json_file:
#     json_file.write(json_data)


# # Define font size
# font_size = 20

# # Set font properties
# font_prop = font_manager.FontProperties(size=font_size, weight='bold')

# # Plot training and validation loss and accuracy for each model on one graph
# fig, axs = plt.subplots(1, 2, figsize=(30, 20))
# # colors = ['red', 'green', 'blue', 'purple', 'gray','black','brown','black','']
# # linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
# colors = ['red', 'green', 'blue', 'purple', 'cyan', 'orange', 
#     'brown', 'pink', 'gray', 'maroon', 'gold']

# linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), 
#     (0, (5, 1)), (0, (5, 5)), (0, (1, 1)), 
#     (0, (3, 5, 1, 5)), (0, (3, 10, 1, 10)), (0, (7, 3))]
# for i, history in enumerate(history_list):
#     # Plot loss
#     axs[0].plot(history.history['loss'], label=f'{model_names[i].__name__} Train', color=colors[i], linestyle=linestyles[i], linewidth=3)
#     axs[0].plot(history.history['val_loss'], label=f'{model_names[i].__name__} Val', color=colors[i], linestyle='--', linewidth=3)

#     # Plot accuracy
#     axs[1].plot(history.history['accuracy'], label=f'{model_names[i].__name__} Train', color=colors[i], linestyle=linestyles[i], linewidth=3)
#     axs[1].plot(history.history['val_accuracy'], label=f'{model_names[i].__name__} Val', color=colors[i], linestyle='--', linewidth=3)

# # Add labels and legend with increased font size
# axs[0].set_title('Training and Validation Loss', fontproperties=font_prop, fontsize=24)
# axs[0].set_xlabel('Epoch', fontproperties=font_prop, fontsize=20)
# axs[0].set_ylabel('Loss', fontproperties=font_prop, fontsize=20)
# axs[0].legend(prop=font_manager.FontProperties(size=18), loc='center right')

# axs[1].set_title('Training and Validation Accuracy', fontproperties=font_prop, fontsize=24)
# axs[1].set_xlabel('Epoch', fontproperties=font_prop, fontsize=20)
# axs[1].set_ylabel('Accuracy', fontproperties=font_prop, fontsize=20)
# axs[1].legend(prop=font_manager.FontProperties(size=18), loc='center right')

# # Increase tick label size
# axs[0].tick_params(labelsize=font_size)
# axs[1].tick_params(labelsize=font_size)

# # Add grid lines and set background color
# axs[0].grid(True)
# axs[1].grid(True)
# fig.patch.set_facecolor('#f2f2f2')
# plt.tight_layout()

# # Save the figure
# file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_acc_loss_curve_all_models.png"
# plt.savefig(os.path.join(output_dir, file_name), dpi=500)
# tf.keras.backend.clear_session()
