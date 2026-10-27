import os,argparse, json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
from sklearn.decomposition import PCA,IncrementalPCA
from sklearn.model_selection import train_test_split
from operator import truediv
import tensorflow as tf
# Libraries
import time
import json
import os, argparse
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pandas as pd
import numpy as np
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
from keras.layers import ( Dense)
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Activation, Lambda, multiply)
from tensorflow.keras.optimizers import Adam
from utils import *

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
# Load the QUH dataset

HSID= "QUH"
HSI, GT, Num_Classes, target_names = LoadHSIData(HSID)
output_dir = os.path.join("HSI_GT")
if not os.path.exists(output_dir):
      os.makedirs(output_dir)

# Display and save the ground truth image
if GT is not None:
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Display ground truth image
    plt.imshow(GT, cmap='nipy_spectral')  # Use 'nipy_spectral' colormap
    plt.title('Ground Truth Image (QUH-Qingyun)')
    plt.axis('off')  # Hide axes for cleaner visualization
    # Save the figure
    file_name = f'{HSID}_ground_truth.png'
    plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0.1)

    # Show the plot
    plt.show()
    
    print(f"Ground truth image saved as '{file_name}'")
else:
    print("Failed to load QUH dataset")
# Check if data is loaded correctly
# if GT is not None:
#     # Display the ground truth image
#     plt.figure(figsize=(8, 8))
#     plt.imshow(GT, cmap='nipy_spectral')  # Use 'jet' colormap for distinct class visualization
#     plt.title('Ground Truth Image (QUH-Qingyun)')
#     plt.colorbar(label='Class Index')
#     plt.axis('off')  # Hide axes for cleaner visualization
#     # Add a legend for class names (optional)
#     # Create a custom legend with class names
#     colors = plt.cm.jet(np.linspace(0, 1, Num_Classes))  # Get colors from the colormap
#     patches = [plt.plot([],[], marker='s', ls="", color=colors[i], label=target_names[i])[0] for i in range(Num_Classes)]
#     plt.legend(patches, target_names, loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")
#     # Show the plot
#     plt.show()\