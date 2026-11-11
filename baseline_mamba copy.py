import os,argparse, json, hashlib
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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Activation, Conv2D, DepthwiseConv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from keras.models import Model, Sequential
from sklearn.manifold import TSNE
# from vit_keras import utils, vit
from keras.utils import to_categorical
from keras_cv_attention_models import attention_layers
import matplotlib.font_manager as font_manager
# from models import WaveMamba,MorpMamba
from utils import *
from sklearn.decomposition import PCA, IncrementalPCA
import warnings
from models.vit import Hybrid_ViT,SSFormer, PyFormer, WaveFormer
from models.vit import wavelet_transform_image
warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser(description="Script Description")
# Add arguments for each parameter
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    choices=["LK", "HC", "UH13", "PU", "PC", "SA","HH","QUH","PI","TH","BS","Fe"],
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
# Evaluation protocol only; does not modify model, optimizer or model.fit().
parser.add_argument("--reference_split", required=True,
                    help="Exact disjoint_spatial_split.npz saved by FKANMamba main.py")
parser.add_argument("--expected_split_protocol", choices=("connected", "blocks"),
                    default="connected", help="Require the selected FKANMamba split protocol")
parser.add_argument("--split_seed", type=int, default=42)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--verify_split_only", action="store_true")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--data_dir", default="./datasets", help="Baseline data directory (original default)")
parser.add_argument("--model", choices=["MorpMamba", "SSFormer", "PyFormer", "Hybrid_ViT", "DSST", "WaveMamba"],
                    default="WaveMamba", help="Model architecture to use")
# Parse the command-line arguments
args = parser.parse_args()

# DataSet Loader Funtion
base_dir = args.data_dir
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
# Match main.py's model seed while keeping the baseline training code unchanged.
tf.keras.utils.set_random_seed(args.seed)
if args.deterministic:
    tf.config.experimental.enable_op_determinism()
adam = tf.keras.optimizers.legacy.Adam(lr = 0.001, decay = 1e-06)
epochs = args.epochs
batch_size = args.batch_size
num_heads=8
out_channels=32
#output_dir = os.path.join(f"baseline_results_out_32_{num_heads}_{batch_size}/{HSID}/{teRatio}/{WS}")
# Keep runs separate by evaluation protocol, split seed and model seed.
output_dir = os.path.join(
    f"baseline_{out_channels}_{num_heads}_{batch_size}", HSID, str(teRatio), str(WS),
    f"split_{args.expected_split_protocol}_{args.split_seed}", f"seed_{args.seed}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# === Exact FKAN evaluation-only helpers (from main.py) ===
def _window_bounds(window_size, buffer=0):
    """Exact source-pixel offsets for any even or odd window size."""
    if window_size < 1 or buffer < 0:
        raise ValueError("window_size must be >= 1, buffer must be >= 0")
    before = window_size // 2 + buffer
    after = window_size - 1 - window_size // 2 + buffer
    return before, after

def footprint_coverage(coords, image_shape, window_size):
    """Fast 2-D difference-array union of all input patch source pixels."""
    before, after = _window_bounds(window_size)
    height, width = image_shape
    result = np.zeros((height + 1, width + 1), dtype=np.int64)
    if len(coords):
        r0 = coords[:, 0] - before
        c0 = coords[:, 1] - before
        r1 = coords[:, 0] + after + 1
        c1 = coords[:, 1] + after + 1
        if (r0.min() < 0 or c0.min() < 0 or
                r1.max() > height or c1.max() > width):
            raise AssertionError("Some patch footprints extend beyond the original image")
        np.add.at(result, (r0, c0), 1)
        np.add.at(result, (r1, c0), -1)
        np.add.at(result, (r0, c1), -1)
        np.add.at(result, (r1, c1), 1)
    covered = result.cumsum(axis=0).cumsum(axis=1)[:height, :width]
    return covered > 0

def verify_disjoint_footprints(coord_groups, image_shape, window_size, owner_map=None):
    """Raise an AssertionError if ANY source pixel appears in multiple splits."""
    used = np.zeros(image_shape, dtype=np.uint8)
    counts = []
    for i, coords in enumerate(coord_groups):
        covered = footprint_coverage(coords, image_shape, window_size)
        overlapping = np.count_nonzero(covered & (used != 0))
        if overlapping:
            raise AssertionError(
                f"Spatial leakage: split {i} overlaps {overlapping} source pixels "
                "from previously checked splits"
            )
        if owner_map is not None and np.any(owner_map[covered] != i):
            raise AssertionError(f"Split {i} uses source pixels owned by another split")
        used[covered] = i + 1
        counts.append(int(np.count_nonzero(covered)))
    print("PASS: zero shared source pixels across train, validation and test patches.")
    print(f"Distinct source pixels covered in each subset: {counts}")
    return True

def reduce_pca_training_only(hsi, train_coords, n_components, method="PCA"):
    """Fit the dimensionality reducer on TRAINING pixels only, then transform scene."""
    if hsi.ndim != 3:
        raise ValueError("hsi must have shape (height, width, bands)")
    if len(train_coords) < n_components or hsi.shape[-1] < n_components:
        raise ValueError("PCA components exceed training pixels or spectral band count")
    fit_data = np.asarray(hsi[train_coords[:, 0], train_coords[:, 1]], dtype=np.float32)
    if method == "PCA":
        reducer = PCA(n_components=n_components, whiten=True, random_state=345)
    elif method == "iPCA":
        reducer = IncrementalPCA(n_components=n_components, whiten=False,
                                 batch_size=max(1024, 5 * n_components))
    else:
        raise ValueError("method must be 'PCA' or 'iPCA'")
    reducer.fit(fit_data)
    flat = hsi.reshape((-1, hsi.shape[-1]))
    transformed = np.empty((len(flat), n_components), dtype=np.float32)
    for offset in range(0, len(flat), 65536):
        transformed[offset:offset + 65536] = reducer.transform(flat[offset:offset + 65536])
    return transformed.reshape(hsi.shape[:2] + (n_components,)), reducer

def extract_patches_at_coords(hsi, gt, coords, window_size, batch_size=4096):
    """Extract only selected labeled patches; never build the entire patch cube."""
    before, after = _window_bounds(window_size)
    if np.any(coords[:, 0] < before) or np.any(coords[:, 1] < before) or \
       np.any(coords[:, 0] + after >= hsi.shape[0]) or \
       np.any(coords[:, 1] + after >= hsi.shape[1]):
        raise ValueError("All patch footprints must lie inside the image")
    patches = np.empty((len(coords), window_size, window_size, hsi.shape[2]),
                       dtype=np.float32)
    steps = np.arange(window_size) - before
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        rr = chunk[:, 0, None, None] + steps[None, :, None]
        cc = chunk[:, 1, None, None] + steps[None, None, :]
        patches[start:start + len(chunk)] = hsi[rr, cc]
    labels = gt[coords[:, 0], coords[:, 1]].astype(np.int64) - 1
    return patches, labels


def load_reference_disjoint_split(filename, gt, window_size, split_seed,
                                  expected_protocol="connected"):
    """Load FKANMamba's ORIGINAL saved centers; reject wrong datasets/protocols/leakage."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"FKANMamba saved split not found: {filename}")
    if expected_protocol not in ("connected", "blocks"):
        raise ValueError("expected_protocol must be connected or blocks")
    with np.load(filename, allow_pickle=False) as ref:
        required = ("train_coords", "validation_coords", "test_coords", "split_map",
                    "owner_map", "window_size", "seed", "block_size", "spatial_buffer")
        missing = [name for name in required if name not in ref.files]
        if missing:
            raise ValueError(f"Invalid saved split; missing keys: {missing}")
        ws, saved_seed = int(ref["window_size"]), int(ref["seed"])
        if (ws, saved_seed) != (window_size, split_seed):
            raise ValueError(f"Saved split WS/seed={ws}/{saved_seed}, requested="
                             f"{window_size}/{split_seed}")
        # An old block-split archive must never be mistaken for a connected split.
        protocol = str(ref["split_protocol"]) if "split_protocol" in ref.files else ""
        expected_tag = ("github_connected_regions_patchsafe_v1" if expected_protocol == "connected"
                        else "spatial_blocks_patchsafe_v1")
        if protocol != expected_tag:
            raise ValueError(f"Wrong or unverified split protocol: {protocol!r}. "
                             f"Expected {expected_tag!r}.")
        if expected_protocol == "connected" and int(ref["block_size"]) != -1:
            raise ValueError("Connected split must have block_size=-1")
        if "gt_sha256" not in ref.files:
            raise ValueError("Saved split lacks ground-truth SHA256; cannot verify dataset identity")
        actual_gt_hash = hashlib.sha256(np.ascontiguousarray(gt).tobytes()).hexdigest()
        if str(ref["gt_sha256"]) != actual_gt_hash:
            raise ValueError("Saved split ground truth differs from baseline dataset (SHA256 mismatch)")

        split_map = np.asarray(ref["split_map"])
        owner_map = np.asarray(ref["owner_map"])
        if split_map.shape != gt.shape or owner_map.shape != gt.shape:
            raise ValueError("Saved split and baseline GT shapes disagree")
        if not np.isin(split_map, [0, 1, 2, 3]).all():
            raise ValueError("Invalid split_map subset IDs")
        coords = []
        for key in required[:3]:
            x = np.asarray(ref[key])
            if x.ndim != 2 or x.shape[1] != 2 or not np.issubdtype(x.dtype, np.integer):
                raise ValueError(f"{key} must be an (N,2) integer coordinate array")
            if not len(x):
                raise ValueError(f"Empty partition: {key}")
            if np.any(x < 0) or np.any(x[:, 0] >= gt.shape[0]) or np.any(x[:, 1] >= gt.shape[1]):
                raise ValueError(f"Out-of-bounds coordinate in {key}")
            if len(np.unique(x, axis=0)) != len(x):
                raise ValueError(f"Duplicate coordinate in {key}")
            if np.any(gt[x[:, 0], x[:, 1]] == 0):
                raise ValueError(f"Unlabeled coordinate in {key}")
            coords.append(x.astype(np.int32, copy=False))
        for idx, x in enumerate(coords, start=1):
            if np.any(split_map[x[:, 0], x[:, 1]] != idx):
                raise ValueError(f"Saved {required[idx-1]} disagrees with split_map")
        if not np.array_equal(np.bincount(split_map.ravel(), minlength=4)[1:4],
                              np.array([len(x) for x in coords])):
            raise ValueError("Split map contains extra/missing partition centers")
        if np.any(split_map[gt == 0] != 0):
            raise ValueError("Split map includes unlabeled image pixels")
        class_ids = np.unique(gt[gt > 0])
        if "class_ids" not in ref.files or not np.array_equal(ref["class_ids"], class_ids):
            raise ValueError("Saved split class IDs do not match baseline GT")
        support = np.array([
            [int(np.count_nonzero(gt[x[:,0], x[:,1]] == cls)) for cls in class_ids]
            for x in coords], dtype=int)
        for i, key in enumerate(("class_train", "class_validation", "class_test")):
            if key not in ref.files or not np.array_equal(ref[key], support[i]):
                raise ValueError(f"Saved {key} disagrees with actual GT class support")
        if expected_protocol == "connected":
            if (support == 0).any():
                raise ValueError("Connected split must contain every class in all three subsets")
            ntrain = int(ref["connected_train_pixels_per_class"])
            nval = int(ref["connected_validation_pixels_per_class"])
            min_test = int(ref["min_test_per_class"])
            if (support[0] != ntrain).any() or (support[1] != nval).any() or (support[2] < min_test).any():
                raise ValueError("Saved connected-region class quotas are not satisfied")
            # Independently verify that each train/validation class forms ONE 8-connected region.
            from scipy.ndimage import label
            for part_name, x in (("training", coords[0]), ("validation", coords[1])):
                labels = gt[x[:, 0], x[:, 1]]
                for cls in class_ids:
                    cls_coords = x[labels == cls]
                    lo = cls_coords.min(axis=0)
                    hi = cls_coords.max(axis=0) + 1
                    mask = np.zeros(tuple(hi - lo), dtype=np.uint8)
                    shift = cls_coords - lo
                    mask[shift[:,0], shift[:,1]] = 1
                    if label(mask, structure=np.ones((3,3), dtype=np.uint8))[1] != 1:
                        raise ValueError(f"{part_name} class {cls} is not 8-connected")
        verify_disjoint_footprints(coords, gt.shape, window_size, owner_map=owner_map)
        print(f"VERIFIED FKAN {expected_protocol} split: WS={ws}, split_seed={saved_seed}, "
              f"train={len(coords[0]):,}, val={len(coords[1]):,}, test={len(coords[2]):,}; "
              "zero cross-partition patch-source overlap.", flush=True)
    return tuple(coords)

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
# Match the FKANMamba spatial protocol, not the original random pixel split.
# The saved split is defined on the ORIGINAL ground truth before PCA/patches.
Tr_xy, Va_xy, Te_xy = load_reference_disjoint_split(
    args.reference_split, GT, WS, args.split_seed, args.expected_split_protocol)
# The archive fingerprint proves each baseline used the identical saved partition.
archive_hash = hashlib.sha256()
with open(args.reference_split, "rb") as split_stream:
    for split_chunk in iter(lambda: split_stream.read(1024 * 1024), b""):
        archive_hash.update(split_chunk)
saved_split_sha256 = archive_hash.hexdigest()
print(f"Reference split file SHA256: {saved_split_sha256}", flush=True)
print(f"Matched FKAN spatial centers: train={len(Tr_xy):,}, "
      f"validation={len(Va_xy):,}, test={len(Te_xy):,}", flush=True)
if args.verify_split_only:
    print("REFERENCE SPLIT VERIFIED: no PCA, model construction or training performed.")
    raise SystemExit(0)
# Same inductive PCA convention as main.py: fit on train CENTER spectra only.
start = time.time()
RDHSI, _pca = reduce_pca_training_only(HSI, Tr_xy, n_components=k, method=DLM)
DL_Time = time.time() - start
# Same 12x12/even-window center alignment and non-overlapping source patches.
Tr, Tr_labels = extract_patches_at_coords(RDHSI, GT, Tr_xy, window_size=WS)
Va, Va_labels = extract_patches_at_coords(RDHSI, GT, Va_xy, window_size=WS)
Te, Te_labels = extract_patches_at_coords(RDHSI, GT, Te_xy, window_size=WS)
TrC, VaC, TeC = Tr_labels, Va_labels, Te_labels
# Reshape train, validation, and test sets
# Tr = Tr.reshape(-1, WS, WS, k, 1)
TrC = to_categorical(TrC)
# Va = Va.reshape(-1, WS, WS, k, 1)
VaC = to_categorical(VaC)
# Te = Te.reshape(-1, WS, WS, k, 1)
TeC = to_categorical(TeC)


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
# MorpMamba Starts

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
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
  
## Spatial-Spectral Feature Enhancement
class MorpSpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(MorpSpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
        self.spatial_gate = Sequential([
            Dense(out_channels),
            Activation('sigmoid')
        ])
        self.spectral_gate = Sequential([
            Dense(out_channels),
            Activation('sigmoid')
        ])

    def call(self, spatial_tokens, spectral_tokens, center_tokens):
        # Spatial enhancement
        spatial_gate_output = self.spatial_gate(center_tokens)
        spatial_enhanced = spatial_tokens * spatial_gate_output[:, :, tf.newaxis]
        # Spectral enhancement
        spectral_gate_output = self.spectral_gate(center_tokens)
        spectral_gate_output = tf.expand_dims(spectral_gate_output, axis=-2)
        spectral_enhanced = spectral_tokens * spectral_gate_output
        return spatial_enhanced, spectral_enhanced
  
## Spatial-Spectral Morphology
class ErosionLayer(Layer):
    def __init__(self, kernel_size, **kwargs):
        super(ErosionLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
    def build(self, input_shape):
        channels = input_shape[-1]
        self.depthwise_conv = []
        for _ in range(channels):
            conv = DepthwiseConv2D(self.kernel_size, padding='same', use_bias=False)
            conv.build((None, None, None, 1))
            conv.set_weights([tf.ones((self.kernel_size, self.kernel_size, 1, 1))])
            self.depthwise_conv.append(conv)
    def call(self, x):
        eroded_channels = []
        for i in range(x.shape[-1]):
            channel = x[..., i:i+1]
            eroded_channel = -self.depthwise_conv[i](channel)
            eroded_channels.append(eroded_channel)
        return tf.concat(eroded_channels, axis=-1)

class DilationLayer(Layer):
    def __init__(self, kernel_size, **kwargs):
        super(DilationLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
    def build(self, input_shape):
        channels = input_shape[-1]
        self.depthwise_conv = []
        for _ in range(channels):
            conv = DepthwiseConv2D(self.kernel_size, padding='same', use_bias=False)
            conv.build((None, None, None, 1))
            conv.set_weights([tf.ones((self.kernel_size, self.kernel_size, 1, 1))])
            self.depthwise_conv.append(conv)
    def call(self, x):
        dilated_channels = []
        for i in range(x.shape[-1]):
            channel = x[..., i:i+1]
            dilated_channel = self.depthwise_conv[i](channel)
            dilated_channels.append(dilated_channel)
        return tf.concat(dilated_channels, axis=-1)

## Spatial-Spectral Token Generation for Spatial-Spectral Morphology
class SpectralSpatialTokenGeneration(Layer):
    def __init__(self, out_channels, kernel_size=5, **kwargs):
        super(SpectralSpatialTokenGeneration, self).__init__(**kwargs)
        self.erosion_layer_spatial = ErosionLayer(kernel_size)
        self.dilation_layer_spatial = DilationLayer(kernel_size)
        self.erosion_layer_spectral = ErosionLayer(kernel_size)
        self.dilation_layer_spectral = DilationLayer(kernel_size)
        self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size=1)
    def call(self, x):
        # Spatial morphology
        eroded_x_spatial = self.erosion_layer_spatial(x)
        dilated_x_spatial = self.dilation_layer_spatial(x)
        # Spectral morphology
        eroded_x_spectral = self.erosion_layer_spectral(tf.transpose(x, [0, 3, 1, 2]))
        dilated_x_spectral = self.dilation_layer_spectral(tf.transpose(x, [0, 3, 1, 2]))
        eroded_x_spectral = tf.transpose(eroded_x_spectral, [0, 2, 3, 1])
        dilated_x_spectral = tf.transpose(dilated_x_spectral, [0, 2, 3, 1])
        # Combine results
        combined_Spatial = self.conv(tf.concat([eroded_x_spatial, dilated_x_spatial], axis=-1))
        combined_Spectral = self.conv(tf.concat([eroded_x_spectral, dilated_x_spectral], axis=-1))
        return combined_Spatial, combined_Spectral

## State-Space Model
class StateSpaceModel(tf.keras.layers.Layer):
    def __init__(self, state_dim, **kwargs):  # Add 'self'
        super(StateSpaceModel, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.state_transition = Dense(state_dim)
        self.state_update = Dense(state_dim)
    def call(self, x):
        state = tf.zeros([tf.shape(x)[0], self.state_dim])
        for t in range(tf.shape(x)[1]):
            state = self.state_transition(state) + self.state_update(x[:, t, :])
        return state

## Spatial-Spectral Morpoholgy Mamba for HSIC
class MorpSSMambaModel(tf.keras.Model):
    def __init__(self, out_channels, num_heads, state_dim,Num_Classes, dropout=0.1, **kwargs):
        super(MorpSSMambaModel, self).__init__(**kwargs)
        self.token_generation = SpectralSpatialTokenGeneration(out_channels)
        self.multi_head_attention = MultiHeadAttention(out_channels, num_heads, dropout)
        self.feature_enhancement = MorpSpectralSpatialFeatureEnhancement(out_channels)
        self.state_space_model = StateSpaceModel(state_dim)
        self.dense = Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = Dropout(0.4)
        self.classifier = Dense(Num_Classes, activation='softmax')
    def call(self, x):
        spatial_tokens, spectral_tokens = self.token_generation(x)
        center_tokens = spatial_tokens[:, x.shape[1] // 2, :]
        spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
        attention_output = self.multi_head_attention(spatial_enhanced, spectral_enhanced, spectral_enhanced)
        state_output = self.state_space_model(attention_output)
        dense_output = self.dense(state_output)
        output = self.classifier(dense_output)
        return output
    
def MorpMamba(Tr, batch_size, Num_Classes):
    model = MorpSSMambaModel(out_channels=64, num_heads=num_heads, state_dim=64, Num_Classes=Num_Classes, dropout=0.1)
    _ = model(Tr[:batch_size])
    return model
#MorpMamba Ends
# WaveMamba Starts


# Spatial-Spectral Wavelet Mamba**

class SpectralSpatialWaveTokenGeneration(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialWaveTokenGeneration, self).__init__(**kwargs)
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
        self.token_generation = SpectralSpatialWaveTokenGeneration(out_channels)
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
    model = SSMambaModel(out_channels=32, num_heads=8, state_dim=64, Num_Classes=Num_Classes, dropout=0.1)
    sample_input = Tr[:batch_size]
    _ = model(sample_input)
    return model
# WaveMamba ENds

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
    if model_name == WaveMamba:
        Tr = Tr.reshape(-1, WS, WS, k)
        Va = Va.reshape(-1, WS, WS, k)
        Te = Te.reshape(-1, WS, WS, k)
        model = WaveMamba(Tr, batch_size, Num_Classes)
    elif model_name == MorpMamba:
      # Transform Training/Validation/Test samples
      Tr = Tr.reshape(-1, WS, WS, k)
      Va = Va.reshape(-1, WS, WS, k)
      Te = Te.reshape(-1, WS, WS, k)
      model = MorpMamba(Tr, batch_size, Num_Classes)
    elif model_name == SSFormer:
      # Transform Training/Validation/Test samples
      Tr = Tr.reshape(-1, WS, WS, k)
      Va = Va.reshape(-1, WS, WS, k)
      Te = Te.reshape(-1, WS, WS, k)
      model = SSFormer(WS, k, Num_Classes)
    elif model_name == Hybrid_ViT:
      # Transform Training/Validation/Test samples
      Tr = Tr.reshape(-1, WS, WS, k)
      Va = Va.reshape(-1, WS, WS, k)
      Te = Te.reshape(-1, WS, WS, k)
      model = Hybrid_ViT(WS, k, Num_Classes)
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
    plt.imshow(GT, cmap='nipy_spectral', vmin=0, vmax=Num_Classes,
               interpolation='nearest')
    plt.axis('off')
    file_name = f"{HSID}_{trRatio}_{k}_{WS}_{DLM}_Ground_Truths.png"
    plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)
    # Predictions on held-out TEST centers ONLY: no train/val labels in evaluation.
    outputs = np.zeros(GT.shape, dtype=np.int32)
    outputs[Te_xy[:, 0], Te_xy[:, 1]] = np.argmax(Te_Pred, axis=1) + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(outputs, cmap='nipy_spectral', vmin=0, vmax=Num_Classes,
               interpolation='nearest')
    plt.axis('off')
    file_name = f"{HSID}_{trRatio}_{k}_{WS}_{DLM}_Predicted_TEST_GT.png"
    plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

    return history, Tr_Time,Te_Time, oa, aa,kappa, total_params, parameter_stats, flops


# # Define a list of model names
model_names = [{"SSFormer": SSFormer,
                "Hybrid_ViT": Hybrid_ViT, "DSST": DSST,"MorpMamba": MorpMamba,
                "WaveMamba": WaveMamba}[args.model]] 
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
        "Reference Split": os.path.abspath(args.reference_split),
        "Reference Split SHA256": saved_split_sha256,
        "Split Protocol": args.expected_split_protocol,
        "Split Seed": args.split_seed,
        "Model Seed": args.seed,
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











# import os,argparse, json
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
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))
# import keras
# import pywt
# from tensorflow.keras import layers, Model
# from tensorflow.keras.layers import  Conv1D, MaxPooling1D, Flatten, Reshape
# from keras.layers import (Conv3D, Dense,Input, Reshape,LayerNormalization)
# from tensorflow.keras.layers import (Input, InputSpec, Add, AveragePooling2D, LeakyReLU, Multiply, 
#                                      Conv3D, Conv2D, Conv1D, BatchNormalization,MaxPool2D,Dropout,Activation,
#                                      GlobalAveragePooling2D,
#                                      Layer, Flatten, Reshape)
# from keras.losses import categorical_crossentropy
# import tensorflow as tf
# from tensorflow.keras.layers import Layer, Dense, Dropout, Activation, Conv2D, DepthwiseConv2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import regularizers
# from keras.models import Model, Sequential
# from sklearn.manifold import TSNE
# # from vit_keras import utils, vit
# from keras.utils import to_categorical
# from keras_cv_attention_models import attention_layers
# import matplotlib.font_manager as font_manager
# # from models import WaveMamba,MorpMamba
# from utils import *
# from sklearn.decomposition import PCA, IncrementalPCA
# import warnings
# from models.vit import Hybrid_ViT,SSFormer, PyFormer, WaveFormer
# from models.vit import wavelet_transform_image
# warnings.filterwarnings('ignore')



# parser = argparse.ArgumentParser(description="Script Description")
# # Add arguments for each parameter
# parser.add_argument(
#     "--dataset",
#     "-d",
#     type=str,
#     choices=["LK", "HC", "UH13", "PU", "PC", "SA","HH","QUH","PI","TH","Fe"],
#     default="SA",
#     help="Dataset identifier",
# )
# parser.add_argument("--window_size", "-ws", type=int, default=12, help="Window size")    
# parser.add_argument("--dl_method", "-dlm", type=str, default='PCA', help="Dimension Reduction Method")  # 차원 축소 메소드
# parser.add_argument("--test_ratio", "-te", type=float, default=0.90, help="Test ratio")
# parser.add_argument("--train_ratio", "-tr", type=float, default=0.50, help="Train ratio")
# parser.add_argument(
#     "--validation_ratio", "-vr", type=float, default=0.50, help="Validation ratio"
# )
# parser.add_argument(
#     "--number_of_bands",
#     "-k",
#     type=int,
#     default=15,
#     help="Number of bands principal components for dimensionality reduction",
# )
# parser.add_argument(
#     "--epochs", "-e", type=int, default=5, help="Number of epochs for training"
# )
# parser.add_argument(
#     "--batch_size", "-bs", type=int, default=56, help="Batch size for training"
# )
# # Evaluation protocol only; does not modify model, optimizer or model.fit().
# parser.add_argument("--reference_split", required=True,
#                     help="Exact disjoint_spatial_split.npz saved by FKANMamba main.py")
# parser.add_argument("--split_seed", type=int, default=42)
# parser.add_argument("--seed", type=int, default=42)
# parser.add_argument("--verify_split_only", action="store_true")
# parser.add_argument("--deterministic", action="store_true")
# parser.add_argument("--data_dir", default="./datasets", help="Baseline data directory (original default)")
# parser.add_argument("--model", choices=["MorpMamba", "SSFormer", "PyFormer", "Hybrid_ViT", "DSST", "WaveMamba"],
#                     default="WaveMamba", help="Model architecture to use")
# # Parse the command-line arguments
# args = parser.parse_args()

# # DataSet Loader Funtion
# base_dir = args.data_dir
# def LoadHSIData(method):
#     '''
#     데이터셋 이름을 받아서 데이터 로드하는 함수수
#     method: str : 'LK', 'HC', 'UH13', 'PU', 'PC', 'SA', 'BS', 'HH', 'QUH', 'PI', 'TH'
#     dataset name
#     return: HSI, GT, Num_Classes, target_names
#     데이터, 정답, 클래스 수, 클래스 이름
#     '''
    
#     data_path = os.path.join(os.getcwd(),f'{base_dir}')
#     if method == 'LK':
#         ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
#         HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
#         GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt'))['WHU_Hi_LongKou_gt']
#         Num_Classes = 9
#         target_names = ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',       # class name list
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
        
#     elif method == 'BS':
#         HSI = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
#         GT = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
#         Num_Classes = 14
#         target_names = ['Water', 'Hippo Grass', 'Floodplain Grasses 1', 'Floodplain Grasses 2',
#                         'Reeds 1', 'Riparian', 'Firescar 2', 'Island Interior', 'Woodlands',
#                         'Acacia Shrublands', 'Acacia Grasslands', 'Short Mopane', 'Mixed Mopane', 'Exposed Soils']
     
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
        
#     elif method == 'Fe':
#         HSI = sio.loadmat(os.path.join(data_path, 'fei2.mat'))['Y_whole_band']
#         GT = sio.loadmat(os.path.join(data_path, 'FeiC_gt.mat'))['patch_gt']
#         Num_Classes =   20
#         target_names = ['Crops 1', 'Bare Soil', 'Shrubs', 'Playground track', 'Road', 'Trees',
#                         'Crops 2', 'Shadow', 'Concrete floor', 'Concrete roof', 'Caigang watt roof',
#                         'Cars', 'Asphalt roof ', 'Grass', 'Glazed tile roof', 'Clay tile roof ',
#                         'Special material', 'Piazza area ', 'White tile floor', 'Black tile floor']
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
# # Match main.py's model seed while keeping the baseline training code unchanged.
# tf.keras.utils.set_random_seed(args.seed)
# if args.deterministic:
#     tf.config.experimental.enable_op_determinism()
# adam = tf.keras.optimizers.legacy.Adam(lr = 0.001, decay = 1e-06)
# epochs = args.epochs
# batch_size = args.batch_size
# num_heads=8
# out_channels=32
# #output_dir = os.path.join(f"baseline_results_out_32_{num_heads}_{batch_size}/{HSID}/{teRatio}/{WS}")
# output_dir = os.path.join(f"baseline_{out_channels}_{num_heads}_{batch_size}/{HSID}/{teRatio}/{WS}")
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
    
# # === Exact FKAN evaluation-only helpers (from main.py) ===
# def _window_bounds(window_size, buffer=0):
#     """Exact source-pixel offsets for any even or odd window size."""
#     if window_size < 1 or buffer < 0:
#         raise ValueError("window_size must be >= 1, buffer must be >= 0")
#     before = window_size // 2 + buffer
#     after = window_size - 1 - window_size // 2 + buffer
#     return before, after

# def footprint_coverage(coords, image_shape, window_size):
#     """Fast 2-D difference-array union of all input patch source pixels."""
#     before, after = _window_bounds(window_size)
#     height, width = image_shape
#     result = np.zeros((height + 1, width + 1), dtype=np.int64)
#     if len(coords):
#         r0 = coords[:, 0] - before
#         c0 = coords[:, 1] - before
#         r1 = coords[:, 0] + after + 1
#         c1 = coords[:, 1] + after + 1
#         if (r0.min() < 0 or c0.min() < 0 or
#                 r1.max() > height or c1.max() > width):
#             raise AssertionError("Some patch footprints extend beyond the original image")
#         np.add.at(result, (r0, c0), 1)
#         np.add.at(result, (r1, c0), -1)
#         np.add.at(result, (r0, c1), -1)
#         np.add.at(result, (r1, c1), 1)
#     covered = result.cumsum(axis=0).cumsum(axis=1)[:height, :width]
#     return covered > 0

# def verify_disjoint_footprints(coord_groups, image_shape, window_size, owner_map=None):
#     """Raise an AssertionError if ANY source pixel appears in multiple splits."""
#     used = np.zeros(image_shape, dtype=np.uint8)
#     counts = []
#     for i, coords in enumerate(coord_groups):
#         covered = footprint_coverage(coords, image_shape, window_size)
#         overlapping = np.count_nonzero(covered & (used != 0))
#         if overlapping:
#             raise AssertionError(
#                 f"Spatial leakage: split {i} overlaps {overlapping} source pixels "
#                 "from previously checked splits"
#             )
#         if owner_map is not None and np.any(owner_map[covered] != i):
#             raise AssertionError(f"Split {i} uses source pixels owned by another split")
#         used[covered] = i + 1
#         counts.append(int(np.count_nonzero(covered)))
#     print("PASS: zero shared source pixels across train, validation and test patches.")
#     print(f"Distinct source pixels covered in each subset: {counts}")
#     return True

# def reduce_pca_training_only(hsi, train_coords, n_components, method="PCA"):
#     """Fit the dimensionality reducer on TRAINING pixels only, then transform scene."""
#     if hsi.ndim != 3:
#         raise ValueError("hsi must have shape (height, width, bands)")
#     if len(train_coords) < n_components or hsi.shape[-1] < n_components:
#         raise ValueError("PCA components exceed training pixels or spectral band count")
#     fit_data = np.asarray(hsi[train_coords[:, 0], train_coords[:, 1]], dtype=np.float32)
#     if method == "PCA":
#         reducer = PCA(n_components=n_components, whiten=True, random_state=345)
#     elif method == "iPCA":
#         reducer = IncrementalPCA(n_components=n_components, whiten=False,
#                                  batch_size=max(1024, 5 * n_components))
#     else:
#         raise ValueError("method must be 'PCA' or 'iPCA'")
#     reducer.fit(fit_data)
#     flat = hsi.reshape((-1, hsi.shape[-1]))
#     transformed = np.empty((len(flat), n_components), dtype=np.float32)
#     for offset in range(0, len(flat), 65536):
#         transformed[offset:offset + 65536] = reducer.transform(flat[offset:offset + 65536])
#     return transformed.reshape(hsi.shape[:2] + (n_components,)), reducer

# def extract_patches_at_coords(hsi, gt, coords, window_size, batch_size=4096):
#     """Extract only selected labeled patches; never build the entire patch cube."""
#     before, after = _window_bounds(window_size)
#     if np.any(coords[:, 0] < before) or np.any(coords[:, 1] < before) or \
#        np.any(coords[:, 0] + after >= hsi.shape[0]) or \
#        np.any(coords[:, 1] + after >= hsi.shape[1]):
#         raise ValueError("All patch footprints must lie inside the image")
#     patches = np.empty((len(coords), window_size, window_size, hsi.shape[2]),
#                        dtype=np.float32)
#     steps = np.arange(window_size) - before
#     for start in range(0, len(coords), batch_size):
#         chunk = coords[start:start + batch_size]
#         rr = chunk[:, 0, None, None] + steps[None, :, None]
#         cc = chunk[:, 1, None, None] + steps[None, None, :]
#         patches[start:start + len(chunk)] = hsi[rr, cc]
#     labels = gt[coords[:, 0], coords[:, 1]].astype(np.int64) - 1
#     return patches, labels


# def load_reference_disjoint_split(filename, gt, window_size, split_seed):
#     """Use FKANMamba's saved coordinates unchanged; fail closed on discrepancies."""
#     if not os.path.isfile(filename):
#         raise FileNotFoundError(f"FKANMamba spatial partition not found: {filename}")
#     with np.load(filename, allow_pickle=False) as ref:
#         required = ("train_coords", "validation_coords", "test_coords", "split_map",
#                     "owner_map", "window_size", "seed", "block_size", "spatial_buffer")
#         absent = [key for key in required if key not in ref.files]
#         if absent:
#             raise ValueError(f"Invalid FKAN split archive; missing: {absent}")
#         ws = int(ref["window_size"])
#         seed = int(ref["seed"])
#         if ws != window_size or seed != split_seed:
#             raise ValueError(f"FKAN split WS/seed = {ws}/{seed}, requested = "
#                              f"{window_size}/{split_seed}")
#         split_map = np.asarray(ref["split_map"])
#         owner_map = np.asarray(ref["owner_map"])
#         if split_map.shape != gt.shape or owner_map.shape != gt.shape:
#             raise ValueError("GT shape differs from FKAN reference split masks")
#         coords = []
#         for key in required[:3]:
#             x = np.asarray(ref[key])
#             if x.ndim != 2 or x.shape[1] != 2 or not np.issubdtype(x.dtype, np.integer):
#                 raise ValueError(f"{key} is not an (N,2) integer coordinate array")
#             if not len(x):
#                 raise ValueError(f"{key} is empty")
#             if np.any(x < 0) or np.any(x[:,0] >= gt.shape[0]) or np.any(x[:,1] >= gt.shape[1]):
#                 raise ValueError(f"{key} contains out-of-image coordinates")
#             if np.unique(x, axis=0).shape[0] != len(x):
#                 raise ValueError(f"{key} contains duplicate coordinates")
#             if np.any(gt[x[:,0],x[:,1]] <= 0):
#                 raise ValueError(f"{key} includes unlabeled positions in baseline GT")
#             coords.append(x.astype(np.int32, copy=False))
#         if any(np.any(split_map[x[:,0], x[:,1]] != idx)
#                for idx, x in enumerate(coords, start=1)):
#             raise ValueError("Saved split_map contradicts saved partition coordinates")
#         if not np.array_equal(
#                 np.bincount(split_map.ravel(), minlength=4)[1:4],
#                 np.array([len(x) for x in coords])):
#             raise ValueError("Saved split_map has extra/missing labeled centers")
#         if np.any(split_map[gt == 0] != 0):
#             raise ValueError("Saved split_map labels pixels absent from baseline GT")
#         print(f"FKAN reference: block_size={int(ref['block_size'])}, "
#               f"spatial_buffer={int(ref['spatial_buffer'])}, "
#               f"split_seed={seed}", flush=True)
#         # Exactly the same source-pixel footprint verifier used in main.py.
#         verify_disjoint_footprints(coords, gt.shape, window_size, owner_map=owner_map)
#     return tuple(coords)


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
# # Match the FKANMamba spatial protocol, not the original random pixel split.
# # The saved split is defined on the ORIGINAL ground truth before PCA/patches.
# Tr_xy, Va_xy, Te_xy = load_reference_disjoint_split(
#     args.reference_split, GT, WS, args.split_seed)
# print(f"Matched FKAN spatial centers: train={len(Tr_xy):,}, "
#       f"validation={len(Va_xy):,}, test={len(Te_xy):,}", flush=True)
# if args.verify_split_only:
#     print("REFERENCE SPLIT VERIFIED: no PCA, model construction or training performed.")
#     raise SystemExit(0)
# # Same inductive PCA convention as main.py: fit on train CENTER spectra only.
# start = time.time()
# RDHSI, _pca = reduce_pca_training_only(HSI, Tr_xy, n_components=k, method=DLM)
# DL_Time = time.time() - start
# # Same 12x12/even-window center alignment and non-overlapping source patches.
# Tr, Tr_labels = extract_patches_at_coords(RDHSI, GT, Tr_xy, window_size=WS)
# Va, Va_labels = extract_patches_at_coords(RDHSI, GT, Va_xy, window_size=WS)
# Te, Te_labels = extract_patches_at_coords(RDHSI, GT, Te_xy, window_size=WS)
# TrC, VaC, TeC = Tr_labels, Va_labels, Te_labels
# # Reshape train, validation, and test sets
# # Tr = Tr.reshape(-1, WS, WS, k, 1)
# TrC = to_categorical(TrC)
# # Va = Va.reshape(-1, WS, WS, k, 1)
# VaC = to_categorical(VaC)
# # Te = Te.reshape(-1, WS, WS, k, 1)
# TeC = to_categorical(TeC)


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
# # MorpMamba Starts

# class MultiHeadAttention(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs):
#         super(MultiHeadAttention, self).__init__(**kwargs)
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
  
# ## Spatial-Spectral Feature Enhancement
# class MorpSpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
#     def __init__(self, out_channels, **kwargs):
#         super(MorpSpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
#         self.spatial_gate = Sequential([
#             Dense(out_channels),
#             Activation('sigmoid')
#         ])
#         self.spectral_gate = Sequential([
#             Dense(out_channels),
#             Activation('sigmoid')
#         ])

#     def call(self, spatial_tokens, spectral_tokens, center_tokens):
#         # Spatial enhancement
#         spatial_gate_output = self.spatial_gate(center_tokens)
#         spatial_enhanced = spatial_tokens * spatial_gate_output[:, :, tf.newaxis]
#         # Spectral enhancement
#         spectral_gate_output = self.spectral_gate(center_tokens)
#         spectral_gate_output = tf.expand_dims(spectral_gate_output, axis=-2)
#         spectral_enhanced = spectral_tokens * spectral_gate_output
#         return spatial_enhanced, spectral_enhanced
  
# ## Spatial-Spectral Morphology
# class ErosionLayer(Layer):
#     def __init__(self, kernel_size, **kwargs):
#         super(ErosionLayer, self).__init__(**kwargs)
#         self.kernel_size = kernel_size
#     def build(self, input_shape):
#         channels = input_shape[-1]
#         self.depthwise_conv = []
#         for _ in range(channels):
#             conv = DepthwiseConv2D(self.kernel_size, padding='same', use_bias=False)
#             conv.build((None, None, None, 1))
#             conv.set_weights([tf.ones((self.kernel_size, self.kernel_size, 1, 1))])
#             self.depthwise_conv.append(conv)
#     def call(self, x):
#         eroded_channels = []
#         for i in range(x.shape[-1]):
#             channel = x[..., i:i+1]
#             eroded_channel = -self.depthwise_conv[i](channel)
#             eroded_channels.append(eroded_channel)
#         return tf.concat(eroded_channels, axis=-1)

# class DilationLayer(Layer):
#     def __init__(self, kernel_size, **kwargs):
#         super(DilationLayer, self).__init__(**kwargs)
#         self.kernel_size = kernel_size
#     def build(self, input_shape):
#         channels = input_shape[-1]
#         self.depthwise_conv = []
#         for _ in range(channels):
#             conv = DepthwiseConv2D(self.kernel_size, padding='same', use_bias=False)
#             conv.build((None, None, None, 1))
#             conv.set_weights([tf.ones((self.kernel_size, self.kernel_size, 1, 1))])
#             self.depthwise_conv.append(conv)
#     def call(self, x):
#         dilated_channels = []
#         for i in range(x.shape[-1]):
#             channel = x[..., i:i+1]
#             dilated_channel = self.depthwise_conv[i](channel)
#             dilated_channels.append(dilated_channel)
#         return tf.concat(dilated_channels, axis=-1)

# ## Spatial-Spectral Token Generation for Spatial-Spectral Morphology
# class SpectralSpatialTokenGeneration(Layer):
#     def __init__(self, out_channels, kernel_size=5, **kwargs):
#         super(SpectralSpatialTokenGeneration, self).__init__(**kwargs)
#         self.erosion_layer_spatial = ErosionLayer(kernel_size)
#         self.dilation_layer_spatial = DilationLayer(kernel_size)
#         self.erosion_layer_spectral = ErosionLayer(kernel_size)
#         self.dilation_layer_spectral = DilationLayer(kernel_size)
#         self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size=1)
#     def call(self, x):
#         # Spatial morphology
#         eroded_x_spatial = self.erosion_layer_spatial(x)
#         dilated_x_spatial = self.dilation_layer_spatial(x)
#         # Spectral morphology
#         eroded_x_spectral = self.erosion_layer_spectral(tf.transpose(x, [0, 3, 1, 2]))
#         dilated_x_spectral = self.dilation_layer_spectral(tf.transpose(x, [0, 3, 1, 2]))
#         eroded_x_spectral = tf.transpose(eroded_x_spectral, [0, 2, 3, 1])
#         dilated_x_spectral = tf.transpose(dilated_x_spectral, [0, 2, 3, 1])
#         # Combine results
#         combined_Spatial = self.conv(tf.concat([eroded_x_spatial, dilated_x_spatial], axis=-1))
#         combined_Spectral = self.conv(tf.concat([eroded_x_spectral, dilated_x_spectral], axis=-1))
#         return combined_Spatial, combined_Spectral

# ## State-Space Model
# class StateSpaceModel(tf.keras.layers.Layer):
#     def __init__(self, state_dim, **kwargs):  # Add 'self'
#         super(StateSpaceModel, self).__init__(**kwargs)
#         self.state_dim = state_dim
#         self.state_transition = Dense(state_dim)
#         self.state_update = Dense(state_dim)
#     def call(self, x):
#         state = tf.zeros([tf.shape(x)[0], self.state_dim])
#         for t in range(tf.shape(x)[1]):
#             state = self.state_transition(state) + self.state_update(x[:, t, :])
#         return state

# ## Spatial-Spectral Morpoholgy Mamba for HSIC
# class MorpSSMambaModel(tf.keras.Model):
#     def __init__(self, out_channels, num_heads, state_dim,Num_Classes, dropout=0.1, **kwargs):
#         super(MorpSSMambaModel, self).__init__(**kwargs)
#         self.token_generation = SpectralSpatialTokenGeneration(out_channels)
#         self.multi_head_attention = MultiHeadAttention(out_channels, num_heads, dropout)
#         self.feature_enhancement = MorpSpectralSpatialFeatureEnhancement(out_channels)
#         self.state_space_model = StateSpaceModel(state_dim)
#         self.dense = Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
#         self.dropout = Dropout(0.4)
#         self.classifier = Dense(Num_Classes, activation='softmax')
#     def call(self, x):
#         spatial_tokens, spectral_tokens = self.token_generation(x)
#         center_tokens = spatial_tokens[:, x.shape[1] // 2, :]
#         spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
#         attention_output = self.multi_head_attention(spatial_enhanced, spectral_enhanced, spectral_enhanced)
#         state_output = self.state_space_model(attention_output)
#         dense_output = self.dense(state_output)
#         output = self.classifier(dense_output)
#         return output
    
# def MorpMamba(Tr, batch_size, Num_Classes):
#     model = MorpSSMambaModel(out_channels=64, num_heads=num_heads, state_dim=64, Num_Classes=Num_Classes, dropout=0.1)
#     _ = model(Tr[:batch_size])
#     return model
# #MorpMamba Ends
# # WaveMamba Starts


# # Spatial-Spectral Wavelet Mamba**

# class SpectralSpatialWaveTokenGeneration(tf.keras.layers.Layer):
#     def __init__(self, out_channels, **kwargs):
#         super(SpectralSpatialWaveTokenGeneration, self).__init__(**kwargs)
#         self.spatial_tokens = tf.keras.layers.Dense(out_channels)
#         self.spectral_tokens = tf.keras.layers.Dense(out_channels)

#     def call(self, x):
#         B, H, W, C = x.shape
#         # Use tf.shape to handle dynamic batch dimension
#         spatial_tokens = self.spatial_tokens(tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [tf.shape(x)[0], H * W, C]))
#         spectral_tokens = self.spectral_tokens(tf.reshape(tf.transpose(x, [0, 1, 2, 3]), [tf.shape(x)[0], H * W, C]))
#         return spatial_tokens,spectral_tokens

# class SpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
#     def __init__(self, out_channels, **kwargs):  # Add 'self'
#         super(SpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
#         self.spatial_gate = tf.keras.Sequential([
#             tf.keras.layers.Dense(out_channels),
#             tf.keras.layers.Activation('sigmoid'),
#             tf.keras.layers.Reshape((1, out_channels))
#         ])
#         self.spectral_gate = tf.keras.Sequential([
#             tf.keras.layers.Dense(out_channels),
#             tf.keras.layers.Activation('sigmoid'),
#             tf.keras.layers.Reshape((1, out_channels))
#         ])

#     def call(self, spatial_tokens, spectral_tokens, center_tokens):
#         spatial_enhanced = spatial_tokens * self.spatial_gate(center_tokens)
#         spectral_enhanced = spectral_tokens * self.spectral_gate(center_tokens)
#         return spatial_enhanced, spectral_enhanced

# class StateSpaceMode(tf.keras.layers.Layer):
#     def __init__(self, state_dim, **kwargs):
#         super(StateSpaceMode, self).__init__(**kwargs)
#         self.state_dim = state_dim
#         self.state_transition = tf.keras.layers.Dense(units=state_dim, activation="relu")
#         self.state_update = tf.keras.layers.Dense(units=state_dim, activation="relu")

#     def call(self, x):
#         state = tf.zeros([tf.shape(x)[0], self.state_dim])
#         for t in range(tf.shape(x)[1]):
#             # Flatten the input if it has more than 2 dimensions
#             input_t = tf.reshape(x[:, t, :], [tf.shape(x)[0], -1])
#             state = self.state_transition(state) + self.state_update(input_t)
#         return state

# class SSMambaModel(tf.keras.Model):
#     def __init__(self, out_channels, num_heads, state_dim, Num_Classes, dropout=0.1, **kwargs):
#         super(SSMambaModel, self).__init__(**kwargs)
#         self.token_generation = SpectralSpatialWaveTokenGeneration(out_channels)
#         self.feature_enhancement = SpectralSpatialFeatureEnhancement(out_channels)
#         self.state_space_model = StateSpaceMode(state_dim)
#         self.dense = tf.keras.layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))
#         self.dropout = tf.keras.layers.Dropout(0.4)
#         self.classifier = tf.keras.layers.Dense(Num_Classes, activation='softmax')

#     # def apply_wavelet_transform(self, image):
#     #     """Apply a wavelet transform to the input tensor."""
#     #     # Ensure the input is in the form expected by pywt
#     #     image_np = tf.make_ndarray(tf.make_tensor_proto(image))
#     #     """wavelet = 'haar', 'db1', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'"""
#     #     # Apply wavelet transform
#     #     coeffs = pywt.dwt2(image_np, 'coif1')
#     #     cA, (cH, cV, cD) = coeffs
#     #     # Combine coefficients (this example uses a simple concatenation)
#     #     transformed = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])
#     #     # Reshape to maintain 3D shape
#     #     transformed_shape = (image_np.shape[0], image_np.shape[1], -1)
#     #     transformed = transformed.reshape(transformed_shape)
#     #     return tf.convert_to_tensor(transformed, dtype=tf.float32)
#     def apply_wavelet_transform(self, image, wavelet_name='bior1.3'):
#         """'haar', 'db1', 'db2', 'coif1', 'bior1.3'"""
#         """Apply a wavelet transform to the input tensor."""
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
#         output = self.classifier(state_output)
#         return output

# def WaveMamba(Tr, batch_size, Num_Classes):
#     model = SSMambaModel(out_channels=32, num_heads=8, state_dim=64, Num_Classes=Num_Classes, dropout=0.1)
#     sample_input = Tr[:batch_size]
#     _ = model(sample_input)
#     return model
# # WaveMamba ENds

# #Model Complexity Calculation
# def count_model_parameters(model):
#     """Count parameters using the same Keras API used by the GitHub baseline.

#     model.count_params() is the canonical total. Trainable and non-trainable
#     counts are reported separately so the result is auditable.
#     """
#     total = int(model.count_params())
#     trainable = int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
#     non_trainable = int(sum(np.prod(weight.shape) for weight in model.non_trainable_weights))

#     if total != trainable + non_trainable:
#         raise RuntimeError(
#             "Keras parameter accounting mismatch: "
#             f"total={total}, trainable={trainable}, non_trainable={non_trainable}."
#         )

#     return {
#         "total": total,
#         "trainable": trainable,
#         "non_trainable": non_trainable,
#     }


# def get_flops(model, input_shape):
#     """Profile TensorFlow graph FLOPs for one input sample.

#     This replaces the previous hand-written layer counter, which double-counted
#     nested layers and missed/under-counted attention, Dense token positions,
#     Conv3D operations, and raw TensorFlow matmuls.

#     Returns None when TensorFlow cannot freeze/profile a model graph.
#     """
#     if len(input_shape) != 3:
#         raise ValueError(f"Expected input_shape=(H, W, C), received {input_shape}.")

#     try:
#         from tensorflow.python.framework.convert_to_constants import (
#             convert_variables_to_constants_v2,
#         )

#         input_spec = tf.TensorSpec(
#             shape=(1, *tuple(input_shape)),
#             dtype=tf.float32,
#             name="model_input",
#         )
#         concrete_function = tf.function(
#             lambda x: model(x, training=False)
#         ).get_concrete_function(input_spec)
#         frozen_function = convert_variables_to_constants_v2(concrete_function)

#         options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#         options["output"] = "none"
#         profile = tf.compat.v1.profiler.profile(
#             graph=frozen_function.graph,
#             options=options,
#         )
#         return int(profile.total_float_ops) if profile is not None else None
#     except Exception as exc:
#         warnings.warn(f"FLOP profiling failed: {exc}")
#         return None


# def compute_flops(model, input_shape):
#     return get_flops(model, input_shape)


# # Train and Evaluate Model
# def train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, output_dir, HSID, teRatio, k, WS, DLM, RDHSI, GT,Num_Classes,batch_size,epochs, target_names):
#     # Create the output directory if it doesn't exist
#     output_dir = os.path.join(output_dir, "") + model_name.__name__
#     os.makedirs(output_dir, exist_ok=True)
#     print(model_name.__name__)
#     input_shape = (WS, WS, k)  # Define the input shape for the model
#     if model_name == WaveMamba:
#         Tr = Tr.reshape(-1, WS, WS, k)
#         Va = Va.reshape(-1, WS, WS, k)
#         Te = Te.reshape(-1, WS, WS, k)
#         model = WaveMamba(Tr, batch_size, Num_Classes)
#     elif model_name == MorpMamba:
#       # Transform Training/Validation/Test samples
#       Tr = Tr.reshape(-1, WS, WS, k)
#       Va = Va.reshape(-1, WS, WS, k)
#       Te = Te.reshape(-1, WS, WS, k)
#       model = MorpMamba(Tr, batch_size, Num_Classes)
#     elif model_name == SSFormer:
#       # Transform Training/Validation/Test samples
#       Tr = Tr.reshape(-1, WS, WS, k)
#       Va = Va.reshape(-1, WS, WS, k)
#       Te = Te.reshape(-1, WS, WS, k)
#       model = SSFormer(WS, k, Num_Classes)
#     elif model_name == Hybrid_ViT:
#       # Transform Training/Validation/Test samples
#       Tr = Tr.reshape(-1, WS, WS, k)
#       Va = Va.reshape(-1, WS, WS, k)
#       Te = Te.reshape(-1, WS, WS, k)
#       model = Hybrid_ViT(WS, k, Num_Classes)
#     else:
#         model = model_name(WS,k,Num_Classes)
#     # Calling Custom Model
#     #model = model_name(WS, k, Num_Classes)
#     file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_model_summary_of_{model_name.__name__}.txt"
#     # Assuming `model` is your Keras model
#     total_params = model.count_params()
#     parameter_stats = count_model_parameters(model)
#     # flops = get_flops(model)
#     flops = get_flops(model, input_shape=input_shape)
#     with open(os.path.join(output_dir, file_name),'w') as fh:
#         # Pass the file handle in as a lambda function to make it callable
#         model.summary(print_fn=lambda x: fh.write(x + '\n'))
#         fh.write(f"\nTotal Parameters: {total_params}\n")
#         fh.write(f"\nTotal Parameters: {parameter_stats['total']}\n")
#         fh.write(f"Trainable Parameters: {parameter_stats['trainable']}\n")
#         fh.write(f"Non-trainable Parameters: {parameter_stats['non_trainable']}\n")
#         fh.write("FLOPs (batch=1): "
#             + (str(flops) if flops is not None else "NA")
#             + "\n"
#         )
        
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
#     # PLot the Ground Truth value
#     # outputs = GT_Plot(CRDHSI, GT, model, WS, k)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(GT, cmap='nipy_spectral', vmin=0, vmax=Num_Classes,
#                interpolation='nearest')
#     plt.axis('off')
#     file_name = f"{HSID}_{trRatio}_{k}_{WS}_{DLM}_Ground_Truths.png"
#     plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)
#     # Predictions on held-out TEST centers ONLY: no train/val labels in evaluation.
#     outputs = np.zeros(GT.shape, dtype=np.int32)
#     outputs[Te_xy[:, 0], Te_xy[:, 1]] = np.argmax(Te_Pred, axis=1) + 1
#     plt.figure(figsize=(10, 10))
#     plt.imshow(outputs, cmap='nipy_spectral', vmin=0, vmax=Num_Classes,
#                interpolation='nearest')
#     plt.axis('off')
#     file_name = f"{HSID}_{trRatio}_{k}_{WS}_{DLM}_Predicted_TEST_GT.png"
#     plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png',
#                 bbox_inches='tight', pad_inches=0)
#     plt.close()

#     return history, Tr_Time,Te_Time, oa, aa,kappa, total_params, parameter_stats, flops


# # # Define a list of model names
# model_names = [{"SSFormer": SSFormer,
#                 "Hybrid_ViT": Hybrid_ViT, "DSST": DSST,"MorpMamba": MorpMamba,
#                 "WaveMamba": WaveMamba}[args.model]] 
# # Loop over the model names and run train_and_evaluate_model() for each model
# all_results = {}
# history_list = []
# Tr_Time_list= []
# Te_Time_list= []
# for model_name in model_names:
#     history, Tr_Time,Te_Time, oa, aa,kappa, total_params, parameter_stats, flops = train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, output_dir, HSID,
#                                                                                     teRatio, k, WS, DLM, RDHSI, GT, Num_Classes,
#                                                                                     batch_size, epochs, target_names)
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
#         "Total Param": total_params,
#         "Total Parameters": parameter_stats["total"],
#         "Trainable Parameters": parameter_stats["trainable"],
#         "Non-trainable Parameters": parameter_stats["non_trainable"],
#         "FLOPs (batch=1)": flops,
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










# import os,argparse, json
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
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))
# import keras
# import pywt
# from tensorflow.keras import layers, Model
# from tensorflow.keras.layers import  Conv1D, MaxPooling1D, Flatten, Reshape
# from keras.layers import (Conv3D, Dense,Input, Reshape,LayerNormalization)
# from tensorflow.keras.layers import (Input, InputSpec, Add, AveragePooling2D, LeakyReLU, Multiply, 
#                                      Conv3D, Conv2D, Conv1D, BatchNormalization,MaxPool2D,Dropout,Activation,
#                                      GlobalAveragePooling2D,
#                                      Layer, Flatten, Reshape)
# from keras.losses import categorical_crossentropy
# from tensorflow.keras import regularizers
# from keras.models import Model, Sequential
# from sklearn.manifold import TSNE
# # from vit_keras import utils, vit
# from keras.utils import to_categorical
# from keras_cv_attention_models import attention_layers
# import matplotlib.font_manager as font_manager
# from models import WaveMamba,MorphMamba
# from utils import *
# import warnings
# from models.vit import Hybrid_ViT,SSFormer, PyFormer, WaveFormer
# from models.vit import wavelet_transform_image
# warnings.filterwarnings('ignore')



# parser = argparse.ArgumentParser(description="Script Description")
# # Add arguments for each parameter
# parser.add_argument(
#     "--dataset",
#     "-d",
#     type=str,
#     choices=["LK", "HC", "UH13", "PU", "PC", "SA","HH","QUH","PI","TH","Fe"],
#     default="SA",
#     help="Dataset identifier",
# )
# parser.add_argument("--window_size", "-ws", type=int, default=12, help="Window size")    
# parser.add_argument("--dl_method", "-dlm", type=str, default='PCA', help="Dimension Reduction Method")  # 차원 축소 메소드
# parser.add_argument("--test_ratio", "-te", type=float, default=0.90, help="Test ratio")
# parser.add_argument("--train_ratio", "-tr", type=float, default=0.50, help="Train ratio")
# parser.add_argument(
#     "--validation_ratio", "-vr", type=float, default=0.50, help="Validation ratio"
# )
# parser.add_argument(
#     "--number_of_bands",
#     "-k",
#     type=int,
#     default=15,
#     help="Number of bands principal components for dimensionality reduction",
# )
# parser.add_argument(
#     "--epochs", "-e", type=int, default=5, help="Number of epochs for training"
# )
# parser.add_argument(
#     "--batch_size", "-bs", type=int, default=56, help="Batch size for training"
# )
# # Parse the command-line arguments
# args = parser.parse_args()

# # DataSet Loader Funtion
# base_dir = './datasets'
# def LoadHSIData(method):
#     '''
#     데이터셋 이름을 받아서 데이터 로드하는 함수수
#     method: str : 'LK', 'HC', 'UH13', 'PU', 'PC', 'SA', 'BS', 'HH', 'QUH', 'PI', 'TH'
#     dataset name
#     return: HSI, GT, Num_Classes, target_names
#     데이터, 정답, 클래스 수, 클래스 이름
#     '''
    
#     data_path = os.path.join(os.getcwd(),f'{base_dir}')
#     if method == 'LK':
#         ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
#         HSI = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
#         GT = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt'))['WHU_Hi_LongKou_gt']
#         Num_Classes = 9
#         target_names = ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',       # class name list
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
        
#     elif method == 'BS':
#         HSI = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
#         GT = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
#         Num_Classes = 14
#         target_names = ['Water', 'Hippo Grass', 'Floodplain Grasses 1', 'Floodplain Grasses 2',
#                         'Reeds 1', 'Riparian', 'Firescar 2', 'Island Interior', 'Woodlands',
#                         'Acacia Shrublands', 'Acacia Grasslands', 'Short Mopane', 'Mixed Mopane', 'Exposed Soils']
     
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
        
#     elif method == 'Fe':
#         HSI = sio.loadmat(os.path.join(data_path, 'fei2.mat'))['Y_whole_band']
#         GT = sio.loadmat(os.path.join(data_path, 'FeiC_gt.mat'))['patch_gt']
#         Num_Classes =   20
#         target_names = ['Crops 1', 'Bare Soil', 'Shrubs', 'Playground track', 'Road', 'Trees',
#                         'Crops 2', 'Shadow', 'Concrete floor', 'Concrete roof', 'Caigang watt roof',
#                         'Cars', 'Asphalt roof ', 'Grass', 'Glazed tile roof', 'Clay tile roof ',
#                         'Special material', 'Piazza area ', 'White tile floor', 'Black tile floor']
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
# output_dir = os.path.join(f"baseline_{out_channels}_{num_heads}_{batch_size}/{HSID}/{teRatio}/{WS}")
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

# #Model Complexity Calculation
# def count_model_parameters(model):
#     """Count parameters using the same Keras API used by the GitHub baseline.

#     model.count_params() is the canonical total. Trainable and non-trainable
#     counts are reported separately so the result is auditable.
#     """
#     total = int(model.count_params())
#     trainable = int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
#     non_trainable = int(sum(np.prod(weight.shape) for weight in model.non_trainable_weights))

#     if total != trainable + non_trainable:
#         raise RuntimeError(
#             "Keras parameter accounting mismatch: "
#             f"total={total}, trainable={trainable}, non_trainable={non_trainable}."
#         )

#     return {
#         "total": total,
#         "trainable": trainable,
#         "non_trainable": non_trainable,
#     }


# def get_flops(model, input_shape):
#     """Profile TensorFlow graph FLOPs for one input sample.

#     This replaces the previous hand-written layer counter, which double-counted
#     nested layers and missed/under-counted attention, Dense token positions,
#     Conv3D operations, and raw TensorFlow matmuls.

#     Returns None when TensorFlow cannot freeze/profile a model graph.
#     """
#     if len(input_shape) != 3:
#         raise ValueError(f"Expected input_shape=(H, W, C), received {input_shape}.")

#     try:
#         from tensorflow.python.framework.convert_to_constants import (
#             convert_variables_to_constants_v2,
#         )

#         input_spec = tf.TensorSpec(
#             shape=(1, *tuple(input_shape)),
#             dtype=tf.float32,
#             name="model_input",
#         )
#         concrete_function = tf.function(
#             lambda x: model(x, training=False)
#         ).get_concrete_function(input_spec)
#         frozen_function = convert_variables_to_constants_v2(concrete_function)

#         options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#         options["output"] = "none"
#         profile = tf.compat.v1.profiler.profile(
#             graph=frozen_function.graph,
#             options=options,
#         )
#         return int(profile.total_float_ops) if profile is not None else None
#     except Exception as exc:
#         warnings.warn(f"FLOP profiling failed: {exc}")
#         return None


# def compute_flops(model, input_shape):
#     return get_flops(model, input_shape)


# # Train and Evaluate Model
# def train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, output_dir, HSID, teRatio, k, WS, DLM, RDHSI, GT,Num_Classes,batch_size,epochs, target_names):
#     # Create the output directory if it doesn't exist
#     output_dir = os.path.join(output_dir, "") + model_name.__name__
#     os.makedirs(output_dir, exist_ok=True)
#     print(model_name.__name__)
#     input_shape = (WS, WS, k)  # Define the input shape for the model
#     if model_name == WaveMamba:
#         Tr = Tr.reshape(-1, WS, WS, k)
#         Va = Va.reshape(-1, WS, WS, k)
#         Te = Te.reshape(-1, WS, WS, k)
#         model = WaveMamba(Tr, batch_size, Num_Classes)
#     if model_name == WaveFormer:
#       # Transform Training/Validation/Test samples
#       Tr = Tr.reshape(-1, WS, WS, k)
#       Va = Va.reshape(-1, WS, WS, k)
#       Te = Te.reshape(-1, WS, WS, k)
#       Tr = wavelet_transform_image(Tr)
#       Va = wavelet_transform_image(Va)
#       Te = wavelet_transform_image(Te)
#       model = WaveFormer(WS, k, Num_Classes)
#     elif model_name == SSFormer:
#       # Transform Training/Validation/Test samples
#       Tr = Tr.reshape(-1, WS, WS, k)
#       Va = Va.reshape(-1, WS, WS, k)
#       Te = Te.reshape(-1, WS, WS, k)
#       model = SSFormer(WS, k, Num_Classes)
#     elif model_name == Hybrid_ViT:
#       # Transform Training/Validation/Test samples
#       Tr = Tr.reshape(-1, WS, WS, k)
#       Va = Va.reshape(-1, WS, WS, k)
#       Te = Te.reshape(-1, WS, WS, k)
#       model = Hybrid_ViT(WS, k, Num_Classes)
#     else:
#         model = model_name(WS,k,Num_Classes)
#     # Calling Custom Model
#     #model = model_name(WS, k, Num_Classes)
#     file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_model_summary_of_{model_name.__name__}.txt"
#     # Assuming `model` is your Keras model
#     total_params = model.count_params()
#     parameter_stats = count_model_parameters(model)
#     # flops = get_flops(model)
#     flops = get_flops(model, input_shape=input_shape)
#     with open(os.path.join(output_dir, file_name),'w') as fh:
#         # Pass the file handle in as a lambda function to make it callable
#         model.summary(print_fn=lambda x: fh.write(x + '\n'))
#         fh.write(f"\nTotal Parameters: {total_params}\n")
#         fh.write(f"\nTotal Parameters: {parameter_stats['total']}\n")
#         fh.write(f"Trainable Parameters: {parameter_stats['trainable']}\n")
#         fh.write(f"Non-trainable Parameters: {parameter_stats['non_trainable']}\n")
#         fh.write("FLOPs (batch=1): "
#             + (str(flops) if flops is not None else "NA")
#             + "\n"
#         )
        
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
#     # PLot the Ground Truth value
#     # outputs = GT_Plot(CRDHSI, GT, model, WS, k)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(GT, cmap='nipy_spectral')
#     plt.axis('off')
#     file_name = f"{HSID}_{trRatio}_{k}_{WS}_{DLM}_Ground_Truths.png"
#     plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)
#     # Ground Truths
#     outputs = GT_Plot(CRDHSI, GT, model, WS, k)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(outputs, cmap='nipy_spectral')
#     plt.axis('off')
#     file_name = f"{HSID}_{trRatio}_{k}_{WS}_{DLM}_Ground_Truths.png"
#     plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)

#     return history, Tr_Time,Te_Time, oa, aa,kappa, total_params, parameter_stats, flops

# # # Define a list of model names
# model_names = [WaveFormer] #[DSST] 
# # Loop over the model names and run train_and_evaluate_model() for each model
# all_results = {}
# history_list = []
# Tr_Time_list= []
# Te_Time_list= []
# for model_name in model_names:
#     history, Tr_Time,Te_Time, oa, aa,kappa, total_params, parameter_stats, flops = train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, output_dir, HSID,
#                                                                                     teRatio, k, WS, DLM, RDHSI, GT, Num_Classes,
#                                                                                     batch_size, epochs, target_names)
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
#         "Total Param": total_params,
#         "Total Parameters": parameter_stats["total"],
#         "Trainable Parameters": parameter_stats["trainable"],
#         "Non-trainable Parameters": parameter_stats["non_trainable"],
#         "FLOPs (batch=1)": flops,
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
