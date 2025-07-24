# Libraries
import time
import json
import os, argparse
import pandas as pd
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"]=""  # Set the GPU ID to use
# Matplotlib (visualization)
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.colors import ListedColormap
# Sklearrn,Scipy, seaborn
import scipy.io as sio  # MATLAB data file
import seaborn as sns
from scipy.signal import medfilt    # Scipy signal process library (median filter)
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
from keras import backend as Kb
from keras.optimizers import Adam
# TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation,Reshape

from utils import *
parser = argparse.ArgumentParser(description="Script Description")
# Add arguments for each parameter
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    choices=["LK","HC","UH13","PU","PC", "SA","HH","QUH","PI","TH"],
    default="HC",
    help="Dataset identifier",
)
parser.add_argument("--window_size", "-ws", type=int, default=14, help="Window size")    
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
    "--epochs", "-e", type=int, default=50, help="Number of epochs for training"
)
parser.add_argument(
    "--batch_size", "-bs", type=int, default=56, help="Batch size for training"
)
# Parse the command-line arguments
args = parser.parse_args()

# DataSet Loader Funtion
base_dir = '../hsi_datasets' # Base directory for HSI data
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
    else:
        print("Wrong Choice")

    return HSI, GT, Num_Classes, target_names

# Global Parameters
HSID = args.dataset # "SLA", "IP", "PU", "PC", "SA", "KSC", "BS", "LK", "HH" (difficult to compile), "HC"
DLM = args.dl_method ## "PCA", "iPCA"
WS = args.window_size
# Using the calculated values
trRatio = args.train_ratio
vrRatio = args.validation_ratio
teRatio = args.test_ratio
k = args.number_of_bands
num_heads=8 
epochs = args.epochs
batch_size = args.batch_size
# adam = tf.keras.optimizers.legacy.Adam(learning_rate = 0.001, decay = 1e-06)
adam = tf.keras.optimizers.Adam(learning_rate = 0.001, weight_decay = 1e-06)
# output_dir = os.path.join(f"Spaital_Spectral_Kan_Mamba/{HSID}/window_size/{WS}")
output_dir = os.path.join(f"Cross_Atn_Mamba_{num_heads}_{batch_size}/{HSID}/{WS}")
if not os.path.exists(output_dir):
      os.makedirs(output_dir)

# %%
## Creat Patches for 3D (Spatial-Spectral) Models
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

# %%
HSI, GT, Num_Classes, target_names = LoadHSIData(HSID)  # Load the HSI data
## Reduce the Dimensionality
#if k < HSI.shape[2]:
start = time.time()
RDHSI = DLMethod(DLM, HSI, NC = k)  # Reduce the dimensionality of the HSI data
end = time.time()
DL_Time = end - start
## Create Image Cubes for Model Building
CRDHSI, CGT = ImageCubes(RDHSI, GT, WS = WS)
## Split Train/validation and Test sets
Tr, Va, Te, TrC, VaC, TeC = TrTeSplit(CRDHSI, CGT, trRatio, vrRatio, teRatio)
# Reshape train, validation, and test sets
# Tr = Tr.reshape(-1, WS, WS, k, 1)
TrC = to_categorical(TrC, num_classes=Num_Classes)  # Convert the train labels to categorical (one-hot encoding)
# Va = Va.reshape(-1, WS, WS, k, 1)
VaC = to_categorical(VaC, num_classes=Num_Classes)
# Te = Te.reshape(-1, WS, WS, k, 1)
TeC = to_categorical(TeC, num_classes=Num_Classes)
print(Te.shape)
print(TeC.shape)

# Token Generation
class SpectralSpatialTokenGeneration(tf.keras.layers.Layer):
    '''
    token generation for spectral and spatial features
    fully connected layer for spectral and spatial features
    '''
    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialTokenGeneration, self).__init__(**kwargs)
        self.spatial_tokens = Dense(out_channels)   # pytorch nn.Linear
        self.spectral_tokens = Dense(out_channels)
    def call(self, x):
        B, H, W, C = x.shape
        # Use tf.shape to handle dynamic batch dimension
        spatial_tokens = self.spatial_tokens(tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [tf.shape(x)[0], H * W, C]))
        spectral_tokens = self.spectral_tokens(tf.reshape(tf.transpose(x, [0, 1, 2, 3]), [tf.shape(x)[0], H * W, C]))
        return spatial_tokens, spectral_tokens

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
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

class SpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):  # Add 'self'
        super(SpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
        self.spatial_gate = Sequential([
            Dense(out_channels),
            Activation('sigmoid'),
            # Reshape the output to match spatial_tokens
            Reshape((1, out_channels))
        ])
        self.spectral_gate = Sequential([
            Dense(out_channels),
            Activation('sigmoid'),
            # Reshape the output to match spectral_tokens
            Reshape((1, out_channels))
        ])

    def call(self, spatial_tokens, spectral_tokens, center_tokens):
        spatial_enhanced = spatial_tokens * self.spatial_gate(center_tokens)
        spectral_enhanced = spectral_tokens * self.spectral_gate(center_tokens)
        return spatial_enhanced, spectral_enhanced

class StateSpaceModel(tf.keras.layers.Layer):
    def __init__(self, state_dim, **kwargs):  # Add 'self'
        super(StateSpaceModel, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.state_transition = Dense(state_dim)
        self.state_update = Dense(state_dim)

    def call(self, x):
        state = tf.zeros([tf.shape(x)[0], self.state_dim])
        # print(x.shape)
        for t in range(tf.shape(x)[1]):
            state = self.state_transition(state) + self.state_update(x[:, t, :])
        return state

class CrossAttentionMambaModel(tf.keras.Model):
    def __init__(self, out_channels, num_heads, state_dim, dropout=0.1, **kwargs): # Pass out_channels to __init__
        super(CrossAttentionMambaModel, self).__init__(**kwargs)
        self.token_generation = SpectralSpatialTokenGeneration(out_channels) # Use out_channels here
        self.cross_attention = CrossAttention(out_channels, num_heads, dropout)
        self.feature_enhancement = SpectralSpatialFeatureEnhancement(out_channels)
        self.state_space_model = StateSpaceModel(state_dim)
        self.dense = Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = Dropout(0.4)
        self.classifier = Dense(Num_Classes, activation='softmax')

    def call(self, x):
        spatial_tokens, spectral_tokens = self.token_generation(x)
        center_tokens = spatial_tokens[:, x.shape[1] // 2, :]
        spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
        attention_output = self.cross_attention(spatial_enhanced, spectral_enhanced, spectral_enhanced)
        state_output = self.state_space_model(attention_output)
        output = self.classifier(state_output)
        return output

# 모델 생성 함수
def CrossAttnMamba(Tr, batch_size):
    model = CrossAttentionMambaModel(
        out_channels=32,
        state_dim=64,
        num_heads=num_heads,
    )
    # Build the model by passing a batch of data
    _ = model(Tr[:batch_size])  # Ensures model is built correctly
    return model

# Train and Evaluate Model
def train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, CRDHSI, HSID, teRatio, k, WS, DLM, RDHSI, GT,batch_size,epochs,output_dir):

    print(f'Model Name : {model_name}')
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(output_dir, "") + model_name.__name__
    os.makedirs(output_dir, exist_ok=True)
    print(model_name.__name__)

    # 모델 명이 FastKanMamba인 경우 dataset을 reshape
    model = CrossAttnMamba(Tr, batch_size)

    # model.build(input_shape=(batch_size, WS, WS, k))  # This line is key
    # Model Summary
    file_name = f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_model_summary_of_{model_name.__name__}.txt"
    # Assuming `model` is your Keras model
    total_params = model.count_params()
    with open(os.path.join(output_dir, file_name),'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    # Compiling the model (모델의 loss, optimizer, metrics 설정)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # Training the model
    start = time.time()
    history = model.fit(x=Tr, y=TrC, batch_size=batch_size, epochs=epochs, validation_data=(Va, VaC))
    end = time.time()
    Tr_Time = end - start
    
    # Test Model
    start = time.time()
    Te_Pre = model.predict(Te)
    end = time.time()
    Te_Time = end - start
    # Calculate FLOPs after the model has been called
    # print("Calculating FLOPs...")
    # model_flops = compute_flops(model)
    ## Classification Report for Test Model
    classification,Confusion,OA,Per_Class,AA,Kappa = ClassificationReports(TeC, Te_Pre, target_names,zero_division=0)
    Te_Kappa = round(Kappa, 4)
    #Writing Results in CSV File
    file_name = os.path.join(output_dir,f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_{batch_size}_Classification_Report.csv")
    CSV_file = CSVResults(file_name, classification, Confusion, Tr_Time, Te_Time, DL_Time, Kappa, OA, AA, Per_Class)
    print(classification)
    # files.download(file_name)
    
    # Ground Truths
    outputs = GT_Plot(CRDHSI, GT, model, WS, k)
    plt.figure(figsize=(10, 10))
    plt.imshow(outputs, cmap='nipy_spectral')
    plt.axis('off')
    file_name = f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_{batch_size}_Ground_Truths.png"
    plt.savefig(os.path.join(output_dir, file_name), dpi=500,format='png', bbox_inches='tight', pad_inches=0)
    return history,Tr_Time, Te_Time, OA, AA,Kappa,total_params

model_names = [CrossAttnMamba]
all_results = {}
history_list = []
Tr_Time_list= []
Te_Time_list= []

for model_name in model_names:
    # Train and Evaluate Model
    history, Tr_Time, Te_Time, OA, AA,Kappa,total_params = train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC,
                                        adam, CRDHSI, HSID, teRatio, k, WS,
                                        DLM, RDHSI, GT, batch_size,epochs,output_dir)
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
with open(all_results_output_path, 'w') as json_file:
    json_file.write(json_data)

# Plot training and validation loss and accuracy for each model
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
colors = ['brown','red','blue','gray', 'green', 'purple', 'orange']
for i, history in enumerate(history_list):
    # Plot loss
    axs[0].plot(history.history['loss'], label=f'{model_names[i].__name__} Train', color=colors[i])
    axs[0].plot(history.history['val_loss'], label=f'{model_names[i].__name__} Val', color=colors[i], linestyle='--')
    # Plot accuracy
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