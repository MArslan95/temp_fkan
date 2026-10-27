# Libraries
import time
import json
import os 
os.environ.setdefault("MPLBACKEND", "Agg")  # headless, publication figures
import argparse
import random
import sys
# Select the GPU outside Python with CUDA_VISIBLE_DEVICES=0 (do not override it here).
import numpy as np
# Matplotlib (visualization)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex
from matplotlib.patches import Patch
# Sklearrn,Scipy, seaborn
import scipy.io as sio  # MATLAB data file
# Spectrak to Load HSI Dataset
from operator import truediv
# All spatial-split and reporting helpers live below in this SAME script.
import warnings
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
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
from keras.layers import (Dense)
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Activation, Lambda, multiply)
from tensorflow.keras.optimizers import Adam
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=1e-06)


# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))
# exit(0)
# ============================================================================
# SPATIALLY DISJOINT TRAIN / VALIDATION / TEST (INTEGRATED — NO EXTRA MODULE)
# Ratio convention: train=.50 and val=.50 split the remaining 10% after test=.90.
# Target = 5% train / 5% val / 90% test (actual retained ratios can vary).
# A test/validation PATCH NEVER REUSES source pixels from another subset.
# ============================================================================

def _window_bounds(window_size, buffer=0):
    """Exact source-pixel offsets for any even or odd window size."""
    if window_size < 1 or buffer < 0:
        raise ValueError("window_size must be >= 1, buffer must be >= 0")
    before = window_size // 2 + buffer
    after = window_size - 1 - window_size // 2 + buffer
    return before, after


def _contained_footprints(owner_map, subset_id, window_size, buffer):
    """Return centers whose *entire* patch + safety buffer belongs to subset_id."""
    height, width = owner_map.shape
    before, after = _window_bounds(window_size, buffer)
    side = before + after + 1
    result = np.zeros((height, width), dtype=bool)
    if height < side or width < side:
        return result

    binary = (owner_map == subset_id).astype(np.int32)
    integral = np.pad(binary, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)
    sums = (integral[side:, side:] - integral[:-side, side:]
            - integral[side:, :-side] + integral[:-side, :-side])
    result[before:height - after, before:width - after] = sums == side * side
    return result


def resolve_split_ratios(train_ratio, validation_ratio, test_ratio):
    """Resolve either absolute fractions or original two-stage split settings.

    Absolute: (.70, .10, .20) -> (.70, .10, .20).
    Two-stage: (.50, .50, .90) -> (.05, .05, .90): first hold out
    90% test, then split remaining 10% equally between train and val.
    """
    values = np.asarray([train_ratio, validation_ratio, test_ratio], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0) or np.any(values >= 1):
        raise ValueError("All ratios must be finite and strictly between 0 and 1")
    if np.isclose(values.sum(), 1.0, rtol=0, atol=1e-8):
        return values / values.sum()
    if np.isclose(train_ratio + validation_ratio, 1.0, rtol=0, atol=1e-8):
        remaining = 1.0 - test_ratio
        return np.array([remaining * train_ratio, remaining * validation_ratio,
                         test_ratio], dtype=float)
    raise ValueError("Ratios must either sum to 1 (absolute fractions), or "
                     "train_ratio + validation_ratio must equal 1 "
                     "(fractions of the non-test portion)")


def _core_tile_histogram(gt, active, tile_of_labeled, classes_compact,
                         block_size, window_size, buffer, num_tiles,
                         num_classes):
    """Conservative class counts always retained if a tile owns its interior."""
    height, width = gt.shape
    before, after = _window_bounds(window_size, buffer)
    r, c = active[:, 0], active[:, 1]
    tile_r, tile_c = r // block_size, c // block_size
    r0, c0 = tile_r * block_size, tile_c * block_size
    r1 = np.minimum(r0 + block_size, height)
    c1 = np.minimum(c0 + block_size, width)
    eligible = ((r - r0 >= before) & (c - c0 >= before) &
                (r + after < r1) & (c + after < c1))
    hist = np.zeros((num_tiles, num_classes), dtype=np.int64)
    np.add.at(hist, (tile_of_labeled[eligible], classes_compact[eligible]), 1)
    return hist


def _guided_owners(rng, core_hist, ratio):
    """Prioritize class coverage in train then val before filling the test set."""
    num_tiles, num_classes = core_hist.shape
    owners = np.full(num_tiles, 2, dtype=np.int8)
    core_count = core_hist.sum(axis=1)
    available = core_count > 0
    rarity = 1.0 / np.maximum(1, np.count_nonzero(core_hist > 0, axis=0))
    # Cover training classes first, then validation classes if possible.
    for group in (0, 1):
        missing = np.any(core_hist > 0, axis=0)
        while missing.any() and available.any():
            gains = (core_hist[:, missing] > 0) @ rarity[missing]
            gain_score = gains / np.maximum(core_count, 1)**0.25
            gain_score *= rng.uniform(0.90, 1.10, size=num_tiles)
            gain_score[~available] = -1
            chosen = int(np.argmax(gain_score))
            if gains[chosen] <= 0:
                break
            owners[chosen] = group
            available[chosen] = False
            missing &= core_hist[chosen] == 0
    # Complete the small train/validation quotas without splitting a tile.
    global_total = int(core_count.sum())
    for group in (0, 1):
        target = ratio[group] * global_total
        have = int(core_count[owners == group].sum())
        while have < target and available.any():
            candidates = np.flatnonzero(available)
            cost = np.abs(have + core_count[candidates] - target)
            cost = cost + rng.uniform(0, 0.02, size=len(cost)) * np.maximum(1, target)
            idx = int(candidates[np.argmin(cost)])
            # Do not add a block that moves us farther away than doing nothing.
            if abs(have + core_count[idx] - target) >= abs(have - target):
                break
            owners[idx] = group
            available[idx] = False
            have += core_count[idx]
    return owners


def disjoint_spatial_split(
    gt, window_size, train_ratio=0.70, validation_ratio=0.10,
    test_ratio=0.20, block_size=32, buffer=0, seed=345,
    search_trials=512,
):
    """Partition labeled HSI centers so train/val/test patches share ZERO pixels.

    Ratios may be absolute (sum to 1) or legacy conditional
    train/validation fractions of the non-test portion. They are targets,
    not guarantees, after block quantization and spatial boundary erosion.
    Returns (train_coords, val_coords, test_coords, split_map, owner_map).
    """
    gt = np.asarray(gt)
    if gt.ndim != 2 or not np.issubdtype(gt.dtype, np.integer):
        raise ValueError("gt must be a 2-D array of integer class IDs (0=unlabeled)")
    if np.any(gt < 0):
        raise ValueError("gt must contain only nonnegative class IDs")
    ratios = resolve_split_ratios(train_ratio, validation_ratio, test_ratio)
    if not isinstance(block_size, (int, np.integer)) or block_size < window_size + 2 * buffer:
        raise ValueError("block_size must be an integer >= window_size + 2 * buffer")
    if search_trials < 1:
        raise ValueError("search_trials must be >= 1")

    height, width = gt.shape
    rows = (height + block_size - 1) // block_size
    cols = (width + block_size - 1) // block_size
    num_tiles = rows * cols
    active = np.argwhere(gt > 0)
    if len(active) == 0:
        raise ValueError("gt has no labeled pixels")
    tile_of_labeled = ((active[:, 0] // block_size) * cols
                       + (active[:, 1] // block_size))
    class_labels = np.unique(gt[gt > 0])
    classes_compact = np.searchsorted(class_labels, gt[active[:, 0], active[:, 1]])
    hist = np.zeros((num_tiles, len(class_labels)), dtype=np.int64)
    np.add.at(hist, (tile_of_labeled, classes_compact), 1)
    active_tiles = np.flatnonzero(hist.sum(axis=1))
    if len(active_tiles) < 3:
        raise ValueError(f"Only {len(active_tiles)} labeled spatial blocks: need >= 3. "
                         "Decrease --block_size if the scene has enough spatial extent")
    core_hist = _core_tile_histogram(gt, active, tile_of_labeled,
                                     classes_compact, block_size, window_size,
                                     buffer, num_tiles, len(class_labels))
    if not np.any(core_hist):
        raise ValueError("No labeled pixels survive inside individual spatial blocks. "
                         "Reduce window size/buffer or adjust block_size")
    # Candidate generation uses core-tile counts (guaranteed retained pixels).
    # Final selection rescans actual post-erosion footprints, not raw tile counts.
    rng = np.random.default_rng(seed)
    core_totals = core_hist.sum(axis=0)
    all_total = int(core_totals.sum())
    potential = np.count_nonzero(core_hist > 0, axis=0) >= 3
    # Small owner regions lose a larger fraction of centers to erosion than
    # the much larger contiguous test region. Explore several OVER-ALLOCATIONS
    # of blocks to small splits, then choose on exact post-erosion ratios.
    scales = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25) if ratios[2] >= .65 else (1.0,)
    candidates_by_scale = [[] for _ in scales]
    for trial in range(search_trials):
        scale_id = trial % len(scales)
        scale = scales[scale_id]
        proposal = ratios.copy()
        proposal[:2] *= scale
        proposal[2] = 1.0 - proposal[0] - proposal[1]
        if proposal[2] <= 0:
            continue
        if trial % 4 != 0:
            owners = _guided_owners(rng, core_hist, proposal)
        else:
            owners = rng.choice(3, size=num_tiles, p=proposal).astype(np.int8)
        counts = np.stack([core_hist[owners == j].sum(axis=0) for j in range(3)])
        totals = counts.sum(axis=1)
        if (totals == 0).any():
            continue
        actual = totals / totals.sum()
        per_cls = counts / np.maximum(1, core_totals)[None, :]
        score = (60.0 * np.sum((actual - proposal)**2)
                 + 0.5 * np.mean((per_cls - proposal[:, None])**2)
                 + 1_000.0 * np.sum(counts[0, core_totals > 0] == 0)
                 + 0.2 * np.sum(counts[1:, potential] == 0))
        candidates_by_scale[scale_id].append((float(score), owners))
    candidates = []
    for bucket in candidates_by_scale:
        bucket.sort(key=lambda item: item[0])
        candidates.extend(bucket[:8])
    if not candidates:
        raise ValueError("Unable to allocate three nonempty spatial partitions; "
                         "try a smaller block size or different seed")
    # Score finalists on *eligible centers* after eroding partition boundaries.
    tile_ids = ((np.arange(height)[:, None] // block_size) * cols
                + np.arange(width)[None, :] // block_size)
    best = None
    best_score = np.inf
    for _proxy_score, owners in candidates:
        owner_map = owners[tile_ids]
        masks = []
        counts = []
        for subset_id in range(3):
            mask = _contained_footprints(owner_map, subset_id, window_size, buffer)
            mask &= gt > 0
            masks.append(mask)
            counts.append(np.bincount(np.searchsorted(class_labels, gt[mask]),
                                      minlength=len(class_labels)))
        totals = np.array([c.sum() for c in counts])
        if (totals == 0).any() or np.any(counts[0] == 0):
            continue
        kept_ratios = totals / totals.sum()
        class_props = np.stack(counts) / np.maximum(1, np.stack(counts).sum(axis=0))[None, :]
        score = (60.0 * np.sum((kept_ratios - ratios)**2)
                 + np.mean((class_props - ratios[:, None])**2)
                 + 0.2 * np.sum(np.stack(counts)[1:, potential] == 0))
        if score < best_score:
            best_score = score
            best = (owner_map.copy(), masks)
    if best is None:
        raise ValueError(
            "No candidate retained all GT classes in training with nonempty validation "
            "and test regions AFTER spatial erosion. For a 5/5/90 split, rare classes "
            "may require >5% training. Reduce --block_size or --spatial_buffer, "
            "increase --search_trials, or try another --split_seed. "
            "Do not silently drop training classes."
        )
    owner_map, masks = best
    split_map = np.zeros((height, width), dtype=np.uint8)
    coords = []
    for i, mask in enumerate(masks):
        split_map[mask] = i + 1
        coords.append(np.argwhere(mask).astype(np.int32))
    for name, part in [("validation", coords[1]), ("testing", coords[2])]:
        present = np.unique(gt[part[:, 0], part[:, 1]])
        missing = np.setdiff1d(class_labels, present)
        if len(missing):
            warnings.warn(f"{name} partition has no pixels for classes "
                          f"{missing.tolist()}; class-specific evaluation is undefined",
                          stacklevel=2)
    verify_disjoint_footprints(coords, gt.shape, window_size, owner_map)
    return (*coords, split_map, owner_map)


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


def predict_labeled_map_in_batches(hsi, gt, model, window_size, batch_size=256):
    """Predict the full labeled scene for visualization with constant-size batches."""
    before, after = _window_bounds(window_size)
    padded = np.pad(hsi, ((before, after), (before, after), (0, 0)))
    coords = np.argwhere(gt > 0)
    output = np.zeros(gt.shape, dtype=np.int32)
    steps = np.arange(window_size)
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        rr = chunk[:, 0, None, None] + steps[None, :, None]
        cc = chunk[:, 1, None, None] + steps[None, None, :]
        patches = padded[rr, cc].astype(np.float32, copy=False)
        prediction = model.predict(patches, verbose=0)
        output[chunk[:, 0], chunk[:, 1]] = prediction.argmax(axis=1) + 1
    return output


# ============================================================================
# CLASSIFICATION AND CSV REPORTING (FORMERLY IMPORTED FROM utils.py)
# ============================================================================

def ClassificationReports(TeC, Te_Pred, target_names,zero_division=0):
    '''
    최종 분류 및 혼동 행렬을 위한 클래스 레이블 할당 함수
    TeC: np.array : test data
    Te_Pred: np.array : test prediction
    target_names: list : target names
    zero_division: int : zero division
    
    return
    classification: str : classification report
    confusion: np.array : confusion matrix
    oa: float : overall accuracy
    each_acc: np.array : each accuracy
    aa: float : average accuracy
    kappa: float : kappa coefficient
    '''
    Te_Pred = np.argmax(Te_Pred, axis=1)    # Get the argmax of the test prediction
    # scikit-learn classification report 생성 (precision, recall, F1-score etc.)
    classification = classification_report(np.argmax(TeC, axis=1), Te_Pred,
                                           labels=np.arange(len(target_names)),
                                           target_names=target_names,
                                           zero_division=zero_division)
    oa = accuracy_score(np.argmax(TeC, axis=1), Te_Pred)    # Overall accuracy, OA
    confusion = confusion_matrix(np.argmax(TeC, axis=1), Te_Pred,
                                 labels=np.arange(len(target_names)))   # Confusion matrix
    list_diag = np.diag(confusion) # Get the diagonal of the confusion matrix (각 클래스별 정분류 개수)
    list_raw_sum = np.sum(confusion, axis=1) # Get the sum of the rows of the confusion matrix (각 클래스별 실제 샘플 개수)
    each_acc = np.divide(list_diag, list_raw_sum,
                         out=np.full(len(list_diag), np.nan, dtype=float),
                         where=list_raw_sum != 0)
    aa = float(np.nanmean(each_acc))  # AA only over classes with test support
    if np.any(list_raw_sum == 0):
        warnings.warn("Some classes have zero test support; their accuracy is undefined and excluded from AA")
    kappa = cohen_kappa_score(np.argmax(TeC, axis=1), Te_Pred)  # Kappa coefficient
    return classification, confusion, oa*100, each_acc*100, aa*100, kappa*100


def CSVResults(file_name, classification, confusion, Tr_Time, Te_Time, DL_Time, kappa, oa, aa, each_acc, memory_used=None):
    '''
    결과 csv 파일에 쓰기
    file_name: str : file name
    classification: str : classification report
    confusion: np.array : confusion matrix
    Tr_Time: float : training time
    Te_Time: float : test time
    DL_Time: float : dimensionality reduction time
    kappa: float : kappa coefficient
    oa: float : overall accuracy
    aa: float : average accuracy
    each_acc: np.array : each accuracy
    '''
    classification = str(classification)
    confusion = str(confusion)
    with open(file_name, 'w') as CSV_file:
      CSV_file.write('{} Tr_Time'.format(Tr_Time))
      CSV_file.write('\n')
      CSV_file.write('{} Te_Time'.format(Te_Time))
      CSV_file.write('\n')
      CSV_file.write('{} DL_Time'.format(DL_Time))
      CSV_file.write('\n')
      CSV_file.write('{} Kappa accuracy (%)'.format(kappa))
      CSV_file.write('\n')
      CSV_file.write('{} Overall accuracy (%)'.format(oa))
      CSV_file.write('\n')
      CSV_file.write('{} Average accuracy (%)'.format(aa))
      CSV_file.write('\n')
      if memory_used is not None:
            CSV_file.write('{} Inference Memory Usage (MB)\n'.format(round(memory_used, 2)))
      CSV_file.write('\n')
      CSV_file.write('{}'.format(classification))
      CSV_file.write('\n')
      CSV_file.write('{}'.format(each_acc))
      CSV_file.write('\n')
      CSV_file.write('{}'.format(confusion))
    return CSV_file

# -----------------------------------------------------------------------------
# PAPER VISUALIZATIONS — SAME SIMPLE imshow STYLE AS ORIGINAL main.py.
# ONLY Matplotlib's nipy_spectral palette is used (all class, split, and plots).
# All plots generated in THIS main.py, no extra plotting utility module.
# -----------------------------------------------------------------------------
def save_paper_figure(fig, stem):
    """500-DPI PNG and PDF; plain image axes like the original HSI script."""
    fig.savefig(stem + '.png', dpi=500, format='png',
                bbox_inches='tight', pad_inches=0.03)
    fig.savefig(stem + '.pdf', format='pdf',
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def _hsi_class_palette(num_classes):
    """Class 1..N sample a FIXED nipy_spectral scale; 0 is masked background."""
    cmap = plt.get_cmap('nipy_spectral')
    return [cmap(i / num_classes) for i in range(1, num_classes + 1)]


def _hsi_discrete_colormap(num_classes):
    cmap = plt.get_cmap('nipy_spectral')
    colors = [cmap(i / num_classes) for i in range(num_classes + 1)]
    return (ListedColormap(colors),
            BoundaryNorm(np.arange(-0.5, num_classes + 1.5), len(colors)),
            colors[1:])


def _paper_map_legend(fig, palette, names, include_background=True, x=0.79):
    handles = []
    if include_background:
        handles.append(Patch(facecolor=plt.get_cmap('nipy_spectral')(0),
                             label='0  Unlabeled / not evaluated'))
    handles += [Patch(facecolor=palette[i], label=f'{i+1:02d}  {name}')
                for i, name in enumerate(names)]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(x, 0.50),
               frameon=False, fontsize=8, ncol=1)


def plot_spatial_partition(gt, split_map, dataset, window_size, split_seed, out_dir):
    """0 unlabeled, 1 excluded, 2 training, 3 validation, 4 testing."""
    gt, split_map = np.asarray(gt), np.asarray(split_map)
    if gt.shape != split_map.shape or not np.isin(split_map, [0, 1, 2, 3]).all():
        raise ValueError('Malformed spatial split map')
    view = np.zeros(gt.shape, dtype=np.uint8)
    view[gt > 0] = 1
    for i in (1, 2, 3):
        view[split_map == i] = i + 1
    counts = [int((view == i).sum()) for i in range(5)]
    names = ['Unlabeled', 'Excluded labeled', 'Train', 'Validation', 'Held-out test']
    # Every map color is an exact sample of nipy_spectral.
    cmap = ListedColormap([plt.get_cmap('nipy_spectral')(i / 4) for i in range(5)])
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.imshow(view, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
    ax.axis('off')
    fig.subplots_adjust(right=0.73)
    fig.legend(handles=[Patch(facecolor=cmap(i / 4), label=f'{names[i]} ({counts[i]:,})')
                for i in range(5)], loc='center left', bbox_to_anchor=(0.73, 0.5),
               frameon=False, fontsize=9)
    save_paper_figure(fig, os.path.join(out_dir, 'disjoint_split_map'))
    return counts


def plot_full_predicted_gt_original_style(gt, predicted, out_dir,
                                          dataset, window_size, num_classes):
    """Save FULL labeled-scene GT and prediction maps exactly in the original style.

    Uses the same `imshow(..., cmap='nipy_spectral')` and class-id scale for
    BOTH images, with unlabeled positions remaining zero. Full-scene predictions
    include training, validation, test and boundary-excluded labeled centers:
    they are for VISUALIZATION ONLY, never used to compute the held-out OA/AA.
    """
    gt = np.asarray(gt)
    predicted = np.asarray(predicted)
    if gt.ndim != 2 or predicted.shape != gt.shape:
        raise ValueError("GT and full-scene prediction must have the same 2-D shape")
    if not np.issubdtype(gt.dtype, np.integer) or not np.issubdtype(predicted.dtype, np.integer):
        raise ValueError("GT and predicted class IDs must be integers")
    if np.any((gt < 0) | (gt > num_classes)) or np.any(
            (predicted < 0) | (predicted > num_classes)):
        raise ValueError("Class IDs must be in 0..num_classes")
    if np.any(predicted[gt == 0] != 0):
        raise ValueError("Predicted GT must leave unlabeled locations zero")

    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"{dataset}_ws{window_size}")
    for label, image in (("Ground_Truth", gt), ("Predicted_GT", predicted)):
        # Original main.py style, matched figure sizes and class colors.
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap='nipy_spectral', vmin=0, vmax=num_classes,
                  interpolation='nearest')
        ax.axis('off')
        fig.savefig(f"{prefix}_{label}.png", dpi=500, format='png',
                    bbox_inches='tight', pad_inches=0)
        fig.savefig(f"{prefix}_{label}.pdf", format='pdf',
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    np.savez_compressed(f"{prefix}_full_scene_prediction.npz",
                        ground_truth=gt, predicted_labels=predicted,
                        class_ids=np.arange(1, num_classes + 1),
                        visualization_only=np.array(True))


def plot_heldout_classification_maps(gt, coords, true_labels, pred_labels,
                                     names, out_dir, dataset, window_size, seed):
    """Simple nipy_spectral maps, showing only genuinely evaluated test centers."""
    gt, coords = np.asarray(gt), np.asarray(coords)
    actual = np.asarray(true_labels, dtype=np.int32)
    predicted = np.asarray(pred_labels, dtype=np.int32)
    n = len(names)
    if gt.ndim != 2 or coords.shape != (len(actual), 2) or actual.shape != predicted.shape:
        raise ValueError('Mismatched held-out map shapes')
    if np.any((actual < 1) | (actual > n)) or np.any((predicted < 1) | (predicted > n)):
        raise ValueError('Class IDs must be in 1..num_classes')
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError('Repeated test coordinates')
    if not np.array_equal(gt[coords[:, 0], coords[:, 1]], actual):
        raise ValueError('Test labels and GT coordinates disagree')
    r, c = coords[:, 0], coords[:, 1]
    test_gt = np.zeros(gt.shape, dtype=np.int32)
    test_pred = np.zeros(gt.shape, dtype=np.int32)
    correctness = np.zeros(gt.shape, dtype=np.uint8)
    test_gt[r, c] = actual
    test_pred[r, c] = predicted
    correctness[r, c] = np.where(actual == predicted, 1, 2).astype(np.uint8)
    cmap, norm, _ = _hsi_discrete_colormap(n)

    # The same no-axis imshow used in the original code. Never infer missing labels.
    for suffix, image in [('full_ground_truth', gt),
                          ('test_ground_truth', test_gt),
                          ('test_predictions', test_pred)]:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap=cmap, norm=norm, interpolation='nearest')
        ax.axis('off')
        save_paper_figure(fig, os.path.join(out_dir, f'{dataset}_ws{window_size}_{suffix}'))

    # Compact comparison. Class IDs have IDENTICAL nipy_spectral colors throughout.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    for ax, (image, title) in zip(axes, [(gt, 'Original ground truth'),
                                          (test_gt, 'Held-out test labels'),
                                          (test_pred, 'Held-out test predictions')]):
        ax.imshow(image, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    fig.tight_layout(pad=0.3)
    save_paper_figure(fig, os.path.join(out_dir,
                                        f'{dataset}_ws{window_size}_heldout_classification'))
    # 0 = no held-out test center; not a negative prediction, and not class 1.
    return {'test_ground_truth': test_gt, 'test_predictions': test_pred,
            'correctness': correctness, 'correct_count': int(np.sum(actual == predicted))}


def plot_test_confusion(true_labels, pred_labels, names, out_dir,
                        dataset, window_size, seed):
    """Confusion percentages on the true-class rows. Colormap nipy_spectral."""
    n = len(names)
    cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(1, n + 1))
    support = cm.sum(axis=1)
    matrix = np.divide(cm * 100., support[:, None],
                       out=np.full(cm.shape, np.nan, dtype=float),
                       where=support[:, None] != 0)
    fig, ax = plt.subplots(figsize=(max(7, n*0.52+2), max(6.5, n*0.45+1.6)))
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap='nipy_spectral',
                   vmin=0, vmax=100, interpolation='nearest', aspect='auto')
    ax.set_xticks(range(n), [str(i) for i in range(1, n + 1)])
    ax.set_yticks(range(n), [f'{i+1} {name} (n={support[i]:,})'
                             for i, name in enumerate(names)], fontsize=8)
    ax.set_xlabel('Predicted class ID')
    ax.set_ylabel('True class')
    for i in range(n):
        if support[i] == 0:
            ax.text(i, i, 'N/A', ha='center', va='center', fontsize=7)
        elif matrix[i, i] >= 1:
            ax.text(i, i, f'{matrix[i,i]:.0f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=.037, pad=.02, label='Row-normalized (%)')
    fig.tight_layout()
    save_paper_figure(fig, os.path.join(out_dir,
                                        f'{dataset}_ws{window_size}_test_confusion'))
    np.savetxt(os.path.join(out_dir, f'{dataset}_ws{window_size}_confusion_counts.csv'),
               cm, fmt='%d', delimiter=',')
    return cm


def plot_class_recall(cm, names, out_dir, dataset, window_size, seed):
    support = np.asarray(cm).sum(axis=1)
    recall = np.divide(np.diag(cm)*100., support,
                       out=np.full(len(names), np.nan), where=support != 0)
    fig, ax = plt.subplots(figsize=(10, max(4, len(names)*.35+1.5)))
    y = np.arange(len(names))
    ax.barh(y, np.nan_to_num(recall), color=_hsi_class_palette(len(names)))
    ax.set_yticks(y, [f'{i+1:02d} {name}' for i,name in enumerate(names)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel('Held-out test recall (%)')
    for i, (value, count) in enumerate(zip(recall, support)):
        ax.text(101, i, f'n={count:,}', va='center', fontsize=8)
    fig.tight_layout()
    save_paper_figure(fig, os.path.join(out_dir,
                                        f'{dataset}_ws{window_size}_per_class_recall'))
    return recall


def plot_learning_curves(hist, out_dir, dataset, window_size, seed):
    hist = hist.history if hasattr(hist, 'history') else hist
    epochs = np.arange(1, len(hist['loss']) + 1)
    cmap = plt.get_cmap('nipy_spectral')
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for name, ax, factor in [('loss', axs[0], 1), ('accuracy', axs[1], 100)]:
        ax.plot(epochs, np.asarray(hist[name])*factor, label='Train',
                color=cmap(.32), linewidth=2)
        if 'val_'+name in hist:
            ax.plot(epochs, np.asarray(hist['val_'+name])*factor,
                    label='Validation', color=cmap(.72), linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss' if name == 'loss' else 'Accuracy (%)')
        ax.set_title('Loss' if name == 'loss' else 'Accuracy')
        ax.legend(frameon=False, fontsize=9)
        if len(epochs) <= 12:
            ax.set_xticks(epochs)
    axs[1].set_ylim(0, 100)
    fig.tight_layout()
    save_paper_figure(fig, os.path.join(out_dir,
                                        f'{dataset}_ws{window_size}_learning_curves'))

#Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Script Description")
# Add arguments for each parameter
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    choices=["LK", "HC", "UH13", "PU", "PC", "SA","HH","QUH","PI","TH"],
    default="PI",
    help="Dataset identifier",
)
parser.add_argument("--window_size", "-ws", type=int, default=4, help="Window size")    
parser.add_argument("--dl_method", "-dlm", type=str, default='PCA', help="Dimension Reduction Method")  # 차원 축소 메소드
parser.add_argument("--test_ratio", "-te", type=float, default=0.90, help="Test holdout fraction; accepts legacy 0.50/0.50/0.90")
parser.add_argument("--train_ratio", "-tr", type=float, default=0.50, help="Training fraction of remaining non-test data when train+val=1")
parser.add_argument(
    "--validation_ratio", "-vr", type=float, default=0.50, help="Validation fraction of remaining non-test data when train+val=1"
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
parser.add_argument("--block_size", type=int, default=32,
                    help="Side length of assigned non-overlapping spatial blocks")
parser.add_argument("--spatial_buffer", type=int, default=0,
                    help="Extra source pixels to exclude around each partition boundary")
parser.add_argument("--search_trials", type=int, default=512,
                    help="Random/guided block allocation trials")
parser.add_argument("--seed", type=int, default=345,
                    help="Run seed: random, NumPy, TensorFlow initialization and training; also split unless overridden")
parser.add_argument("--split_seed", type=int, default=None,
                    help="Optional fixed spatial split seed (defaults to --seed)")
parser.add_argument("--deterministic", action="store_true",
                    help="Request TensorFlow deterministic GPU operations (may reduce performance)")
parser.add_argument("--prediction_map", choices=["none", "test", "full"], default="test",
                    help="Map mode: test/full save held-out and full predicted-GT visualizations; none skips maps")
parser.add_argument("--verify_split_only", action="store_true",
                    help="Verify the spatial split and exit without PCA or model training")
# Parse the command-line arguments
args = parser.parse_args()
# Set before importing TensorFlow. Python's hash seed must be exported before
# interpreter startup for full hash reproducibility (e.g. PYTHONHASHSEED=42).
# Seed model initialization, dropout, and fit shuffling before any layer is created.
tf.keras.utils.set_random_seed(args.seed)
if args.deterministic:
    tf.config.experimental.enable_op_determinism()
if args.deterministic:
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
if args.split_seed is None:
    args.split_seed = args.seed
random.seed(args.seed)
np.random.seed(args.seed)
print(f"Seeds: split={args.split_seed} model={args.seed}; deterministic={args.deterministic}")

# DataSet Loader Funtion
base_dir = './hsi_datasets'
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
HSID = args.dataset ## "SLA", "IP", "PU", "PC", "SA", "KSC", "BS", "LK", "HH" (difficult to compile), "HC"
DLM = args.dl_method ## "PCA", "iPCA"
WS = args.window_size
# Using the calculated values
trRatio = args.train_ratio
vrRatio = args.validation_ratio
teRatio = args.test_ratio
effective_ratios = resolve_split_ratios(trRatio, vrRatio, teRatio)
print("Requested ratios (train, val, test):", trRatio, vrRatio, teRatio)
print("Effective TARGET fractions (train, val, test):", effective_ratios.tolist())
k = args.number_of_bands
num_heads=8
out_channels=32
epochs = args.epochs
batch_size = args.batch_size
# output_dir = os.path.join(f"Spaital_Spectral_Kan_Mamba/{HSID}/window_size/{WS}")
output_dir = os.path.join(
    f"KanMamba_{num_heads}_chnal_{out_channels}_bs_{batch_size}",
    HSID, str(WS), "arch_original_dima_ssm_tokenfix", f"split_{args.split_seed}",
    f"seed_{args.seed}")
if not os.path.exists(output_dir):
      os.makedirs(output_dir)

# %%
# Spatially disjoint patch extraction replaces the old ImageCubes function.
HSI, GT, Num_Classes, target_names = LoadHSIData(HSID)  # Load the HSI data
# STEP 1: Spatially assign blocks on the ORIGINAL image, before PCA or patches.
# Block erosion removes all patch centers whose input footprints cross splits.
Tr_xy, Va_xy, Te_xy, split_map, owner_map = disjoint_spatial_split(
    GT, window_size=WS,
    train_ratio=trRatio, validation_ratio=vrRatio, test_ratio=teRatio,
    block_size=args.block_size, buffer=args.spatial_buffer, seed=args.split_seed,
    search_trials=args.search_trials,
)

# Save the exact split so it can be reproduced and independently audited.
np.savez_compressed(
    os.path.join(output_dir, "disjoint_spatial_split.npz"),
    train_coords=Tr_xy, validation_coords=Va_xy, test_coords=Te_xy,
    split_map=split_map, owner_map=owner_map,
    window_size=WS, block_size=args.block_size,
    spatial_buffer=args.spatial_buffer, seed=args.split_seed, model_seed=args.seed,
    requested_ratios=np.array([trRatio, vrRatio, teRatio]),
    effective_ratios=effective_ratios,
)

n_selected = len(Tr_xy) + len(Va_xy) + len(Te_xy)
n_labeled = int(np.count_nonzero(GT))
print(f"Original labeled centers: {n_labeled:,}")
for name, coords in (("Train", Tr_xy), ("Validation", Va_xy), ("Test", Te_xy)):
    print(f"{name}: {len(coords):,} samples ({100 * len(coords)/n_selected:.2f}% of retained)")
print(f"Excluded by spatial disjointness: {n_labeled - n_selected:,} labeled centers")
if n_selected:
    kept = np.array([len(Tr_xy), len(Va_xy), len(Te_xy)]) / n_selected
    print("Achieved fractions AFTER erosion:", np.round(kept, 4).tolist())
    if np.any(np.abs(kept - effective_ratios) > 0.025):
        print("WARNING: retained ratios differ by >2.5 percentage points from targets; "
              "spatially disjoint blocks cannot promise exact split percentages.")
for group_name, coords in (("train", Tr_xy), ("validation", Va_xy), ("test", Te_xy)):
    labels, counts = np.unique(GT[coords[:, 0], coords[:, 1]], return_counts=True)
    print(group_name, "class support:", dict(zip(labels.tolist(), counts.tolist())))
    sparse = {int(label): int(count) for label, count in zip(labels, counts) if count < 20}
    if sparse and group_name != "test":
        warnings.warn(f"{group_name} has fewer than 20 centers for classes {sparse}; "
                      "pre-register how sparse classes are handled before publication")
# verify_disjoint_footprints is also called internally by disjoint_spatial_split.

# Publication split figure: distinguishes unlabeled from excluded labeled centers.
# Colors describe center allocation, NOT all input pixels of the patches.
plot_spatial_partition(GT, split_map, HSID, WS, args.split_seed, output_dir)

if args.verify_split_only:
    print("Split-only verification completed. No PCA or network training performed.")
    raise SystemExit(0)

# STEP 2: PCA is fitted ONLY on training-center pixels (inductive evaluation).
start = time.time()
RDHSI, pca = reduce_pca_training_only(HSI, Tr_xy, n_components=k, method=DLM)
DL_Time = time.time() - start

# STEP 3: Extract only the three selected, non-overlapping patch subsets.
Tr, TrC = extract_patches_at_coords(RDHSI, GT, Tr_xy, window_size=WS)
Va, VaC = extract_patches_at_coords(RDHSI, GT, Va_xy, window_size=WS)
Te, TeC = extract_patches_at_coords(RDHSI, GT, Te_xy, window_size=WS)


TrC = to_categorical(TrC, num_classes=Num_Classes)
VaC = to_categorical(VaC, num_classes=Num_Classes)
TeC = to_categorical(TeC, num_classes=Num_Classes)
print("Test inputs:", Te.shape, "Test targets:", TeC.shape)

# Token Generation
# Patch coordinates follow the SAME convention as extract_patches_at_coords():
# target pixel is at local (H // 2, W // 2), including even-sized windows.
def _target_token_index(height, width):
    """Zero-based row-major index of the labeled target pixel in an HxW patch."""
    return (height // 2) * width + (width // 2)


def _flatten_hsi_tokens(x):
    """Convert (batch, row, column, bands) to row-major (batch, H*W, bands).

    Preserve all spectral bands together for each spatial location. In particular,
    never transpose spatial and spectral dimensions before flattening.
    """
    shape = tf.shape(x)
    return tf.reshape(x, [shape[0], shape[1] * shape[2], shape[3]])


class SpectralSpatialTokenGeneration(tf.keras.layers.Layer):
    """Two independently learned embeddings of spatially aligned pixel spectra.

    Both branches use the SAME row-major pixel order, so token i in each branch
    corresponds to the same input pixel at fusion. Subsequent attention models
    spatial context. No separate spatial convolution is claimed here.
    """
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.spatial_tokens = Dense(out_channels)
        self.spectral_tokens = Dense(out_channels)

    def call(self, x):
        pixel_spectra = _flatten_hsi_tokens(x)
        return self.spatial_tokens(pixel_spectra), self.spectral_tokens(pixel_spectra)


# Fast Kolomgorov Arnold Network
class SplineLinear(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, init_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.init_scale = init_scale
        self.weight = self.add_weight(
            shape=(self.out_features, self.in_features),
            initializer=tf.keras.initializers.GlorotUniform(), #tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=init_scale), #HeNormal(),
            trainable=True,
            regularizer=tf.keras.regularizers.l2(0.001)
        )
    def call(self, inputs):
        return tf.matmul(inputs, tf.transpose(self.weight))
    
    
# Radial Basis Function for Fast KAN
class RadialBasisFunction(tf.keras.layers.Layer):
    def __init__(self, grid_min=-1.0, grid_max=1.0, num_grids=8, denominator=None, **kwargs):
        super().__init__(**kwargs)
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
        self.grid = tf.Variable(
            initial_value=tf.linspace(grid_min, grid_max, num_grids),
            trainable=True,
            name="grid"
        )
    def call(self, inputs):
        expanded_inputs = tf.expand_dims(inputs, axis=-1)
        return tf.exp(-((expanded_inputs - self.grid) / self.denominator) ** 2)


# Fast Kolmogorov Arnold Network Layer
class FastKANLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, grid_min=-1.0, grid_max=1.0, 
                 num_grids=8, use_base_update=True, use_layernorm=True, base_activation=tf.nn.swish,spline_weight_init_scale=0.1, 
                 **kwargs):
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
        # ret = tf.reshape(ret, shape=tf.shape(inputs)) 
        output_shape = tf.concat([tf.shape(inputs)[:-1], [self.output_dim]], axis=0)
        ret = tf.reshape(ret, shape=output_shape)
        if self.use_base_update:
            base = self.base_linear(self.base_activation(inputs))
            ret = ret + base
        return ret


# 멀티 헤드 어텐션을 Fast KAN으로 구현
# Multihead Attention With Fast Kolmogorov Arnold Network
class MultiHeadKANAttention(tf.keras.layers.Layer):
    def __init__(self, 
                 input_dim, num_heads, output_dim, grid_min=-1.0, grid_max=1.0, num_grids=8, use_base_update=False,base_activation=tf.nn.swish, spline_weight_init_scale=0.1, dropout_rate=0.7, 
                 output_activation=None,  # Optional activation for output
                 **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.output_dim = output_dim
        self.all_head_size = self.num_heads * self.head_dim
        
        # Query, Key, and Value projections via FastKANLayer
        self.query_kan = FastKANLayer(input_dim=input_dim,output_dim=self.all_head_size,grid_min=grid_min,grid_max=grid_max, 
            num_grids=num_grids,use_base_update=use_base_update,base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale
        )
        self.key_kan = FastKANLayer(input_dim=input_dim,output_dim=self.all_head_size,grid_min=grid_min,grid_max=grid_max, 
            num_grids=num_grids,use_base_update=use_base_update,base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale
        )
        self.value_kan = FastKANLayer(input_dim=input_dim,output_dim=self.all_head_size,grid_min=grid_min,grid_max=grid_max, 
            num_grids=num_grids,use_base_update=use_base_update,base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale
        )
        # Final output projection
        self.output_projection = tf.keras.layers.Dense(units=self.output_dim) # Better weight initialization
        # Optional output activation
        self.output_activation = tf.keras.layers.Activation(output_activation) if output_activation else None
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
        attention_scores = tf.matmul(Q, K, transpose_b=True)  # Shape: (batch_size, num_heads, seq_len, seq_len)
        d_k = tf.cast(self.head_dim, tf.float32)  # Head dimension
        attention_scores /= tf.math.sqrt(d_k)  # Scale by sqrt(head_dim) for numerical stability
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # Softmax over sequence length
        # Dropout on attention weights
        attention_weights = self.attention_dropout(attention_weights, training=training)
        # Weighted sum of Value embeddings
        attention_output = tf.matmul(attention_weights, V)  # Shape: (batch_size, num_heads, seq_len, head_dim)
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
        
        self.spatial_gate = Sequential([
            FastKANLayer(input_dim=out_channels,output_dim=out_channels,
            grid_min=-1.0,grid_max=1.0,num_grids=8) ,
            Activation('sigmoid'),
        ])

        self.spectral_gate = Sequential([
            # FastKANLayer(input_dim=out_channels, output_dim=out_channels),
            FastKANLayer(input_dim=out_channels,output_dim=out_channels,
             grid_min=-1.0,grid_max=1.0,num_grids=8),
            Activation('sigmoid'),
        ])

    def call(self, spatial_tokens, spectral_tokens, center_tokens):
        spatial_weights = self.spatial_gate(center_tokens)
        spectral_weights = self.spectral_gate(center_tokens)
        spatial_enhanced = spatial_tokens * tf.expand_dims(spatial_weights, axis=1)
        spectral_enhanced = spectral_tokens * tf.expand_dims(spectral_weights, axis=1)
        return spatial_enhanced, spectral_enhanced

# Original fixed-parameter affine StateSpaceModel from DiMA, unmodified.
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
    '''
    token_generation -> feature_enhancement -> multi_head_attention -> concat -> state_space_model -> classifier(Dense)
    HSI 데이터를 받아서 token_generation을 통해 공간과 스펙트럴 토큰을 생성하고 
    spatial_tokens의 중간 토큰을 gating으로 사용하여 
    feature_enhancement를 통해 토큰을 강화하고
    공간과 스펙트럴 토큰 각각에 multi_head_attention을 적용하고
    어텐션 결과를 concat하여 state_space_model을 통해 시계열 모델링을 하고
    ssm의 결과를 classifier(Dense)에 통과시켜서 분류
    '''
    def __init__(self, out_channels, state_dim=64, num_heads=num_heads, num_grids=8, **kwargs):
        super(SSFKSMambaModel, self).__init__(**kwargs)
        # Tokenization
        self.token_generation = SpectralSpatialTokenGeneration(out_channels)
        # Multi-Head Attention
        self.multi_head_attention = MultiHeadKANAttention(input_dim=out_channels, output_dim=out_channels, num_heads=num_heads, grid_min=-1.0,
            grid_max=1.0,num_grids=num_grids,base_activation=tf.nn.swish)
        # Feature Enhancement
        self.feature_enhancement = SpectralSpatialFeatureEnhancement(out_channels)
        # Original unmodified fixed-parameter affine recurrent aggregation from DiMA
        self.state_space_model = StateSpaceModel(state_dim=state_dim)
        # Dense Layers and Classification
        self.dense = tf.keras.layers.Dense(units=out_channels, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.classifier = tf.keras.layers.Dense(Num_Classes, activation='softmax')

    # Forward pass
    def call(self, x, training=False):
        # Step 1: Tokenization
        spatial_tokens, spectral_tokens = self.token_generation(x)
        # Even WS=12 has a target at (6,6): row-major index 6*12+6 = 78.
        # Use runtime dimensions for dynamic batch/image shapes.
        target_index = _target_token_index(tf.shape(x)[1], tf.shape(x)[2])
        center_tokens = tf.gather(spatial_tokens, target_index, axis=1)
        # Step 2: Feature Enhancement
        spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
        # Step 3: Apply Multi-Head Attention
        spatial_attention = self.multi_head_attention(spatial_enhanced,spatial_enhanced,spatial_enhanced)
        spectral_attention = self.multi_head_attention(spectral_enhanced,spectral_enhanced,spectral_enhanced)
        # Step 4: Combine the outputs
        combined_output = tf.concat([spatial_attention, spectral_attention], axis=-1)
        # Step 5: Original DiMA fixed-parameter affine recurrent aggregation
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
# Reproducible analytical MAC accounting for the EXACT two-branch model.
# A multiply-accumulate (MAC) is reported as 2 FLOPs by convention. This
# intentionally excludes RBF exponentials, softmax, normalization, biases,
# activation functions, gating multiplies and any kernel-level overhead.
# Thus 2*MAC is NOT the complete floating-point-operation count.
def estimate_mac_flops(window_size, bands, num_classes, token_dim, heads,
                       state_dim, qkv_grids, gate_grids,
                       qkv_use_base=False, gate_use_base=True):
    """Count executed matrix MACs for one HSI patch, not model layers once.

    The two attention calls SHARE weights but EACH incurs compute. The affine
    recurrence performs its two Dense projections once at EVERY token.
    No tensor tracing, data, or GPU is required for this operation count.
    """
    values = (window_size, bands, num_classes, token_dim, heads,
              state_dim, qkv_grids, gate_grids)
    if not all(isinstance(v, int) and v > 0 for v in values):
        raise ValueError('MAC profile dimensions must be positive integers')
    if token_dim % heads:
        raise ValueError('token_dim must be divisible by heads')
    n = window_size * window_size
    d, s, c = token_dim, state_dim, bands
    # All counted operations below are matrix products only (multiply-adds).
    ops = {
        'token_projections': 2 * n * c * d,
        'gate_spline': 2 * d * gate_grids * d,  # both gates; one center token
        'gate_base': (2 * d * d if gate_use_base else 0),
        'attention_qkv_spline': 2 * 3 * n * d * qkv_grids * d,
        'attention_qkv_base': (2 * 3 * n * d * d if qkv_use_base else 0),
        'attention_scores': 2 * n * n * d,  # QK^T: two attention calls
        'attention_value_mix': 2 * n * n * d,  # softmax(QK^T)V: two calls
        'attention_output_projection': 2 * n * d * d,
        'recurrent_transition': n * s * s,
        'recurrent_input': n * (2 * d) * s,  # concatenated two branches
        'classifier': s * num_classes,
    }
    total = sum(ops.values())
    return {
        'sequence_length': n,
        'mac_breakdown': ops,
        'macs_per_patch': total,
        'mac_equivalent_flops_per_patch': 2 * total,
        'mac_equivalent_gflops_per_patch': 2 * total / 1e9,
        'method': 'Analytical, 1 multiply-accumulate = 2 FLOPs, per patch; '
                  'weights shared across 2 executed attention calls',
        'includes': 'Dense/SplineLinear/QKV/attention score/value matrix '
                    'products; recurrent matrices at all N time steps',
        'excludes': 'RBF exp/basis arithmetic, LayerNorm, softmax, '
                    'sigmoid/swish, biases, gating/additions; NOT total FLOPs',
        'safe_to_claim_exact_total_flops': False,
    }


def estimate_instantiated_model_macs(model, window_size, bands):
    """Read all relevant dimensions from the BUILT model, not hardcoded CLI."""
    token_dim = int(model.token_generation.spatial_tokens.units)
    attn = model.multi_head_attention
    gate = model.feature_enhancement.spatial_gate.layers[0]
    q = attn.query_kan
    state_dim = int(model.state_space_model.state_dim)
    classes = int(model.classifier.units)
    # Abort instead of emitting plausible but invalid counts for a new variant.
    if int(model.token_generation.spectral_tokens.units) != token_dim:
        raise ValueError('Unexpected token projection sizes')
    if any(x.input_dim != token_dim or x.output_dim != token_dim
           for x in (attn.query_kan, attn.key_kan, attn.value_kan)):
        raise ValueError('Unexpected attention projection architecture')
    if int(attn.all_head_size) != token_dim:
        raise ValueError('Attention projections do not have token_dim outputs')
    if any(g.input_dim != token_dim or g.output_dim != token_dim
           for g in (gate, model.feature_enhancement.spectral_gate.layers[0])):
        raise ValueError('Unexpected feature enhancement gate dimensions')
    if int(model.state_space_model.state_update.kernel.shape[0]) != 2 * token_dim:
        raise ValueError('Unexpected recurrent input dimension')
    if int(model.state_space_model.state_transition.kernel.shape[0]) != state_dim:
        raise ValueError('Unexpected state transition dimension')
    if int(model.classifier.kernel.shape[0]) != state_dim:
        raise ValueError('Unexpected classifier dimension')
    return estimate_mac_flops(
        window_size=int(window_size), bands=int(bands), num_classes=classes,
        token_dim=token_dim, heads=int(attn.num_heads), state_dim=state_dim,
        qkv_grids=int(q.rbf.num_grids), gate_grids=int(gate.rbf.num_grids),
        qkv_use_base=bool(q.use_base_update),
        gate_use_base=bool(gate.use_base_update),
    )


def plot_mac_breakdown(profile, out_dir, dataset, window_size):
    """Operation-group comparison using nipy_spectral only."""
    items = profile['mac_breakdown']
    labels, values = list(items), [2 * v / 1e6 for v in items.values()]
    n = len(labels)
    cmap = plt.get_cmap('nipy_spectral')
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    y = np.arange(n)
    ax.barh(y, values, color=[cmap((i + 1) / (n + 1)) for i in range(n)])
    ax.set_yticks(y, [x.replace('_', ' ') for x in labels], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Matrix-operation FLOPs equivalent (million; 2 per MAC)')
    ax.set_title(f'{dataset}, {window_size}x{window_size} patch — executed matrix operations')
    fig.tight_layout()
    save_paper_figure(fig, os.path.join(out_dir,
                      f'{dataset}_ws{window_size}_matrix_flops_breakdown'))

# FLOPs: ONE forward prediction on ONE (WS, WS, k) patch, float32.
# Standard TF graph-profiler count; NOT the previously incorrect hand count.
def profile_model_flops(model, window_size, bands):
    """Return TensorFlow graph-profiled float operations per sample.

    Caveat: TF's static graph profiler can miss Exp/softmax and operations
    inside While/StatelessWhile (including recurrent loops). Such numbers
    are graph-profiler estimates, NOT exact hardware FLOP measurements.
    """
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    @tf.function(input_signature=[tf.TensorSpec(
        shape=[1, window_size, window_size, bands], dtype=tf.float32)])
    def inference(patch):
        return model(patch, training=False)

    concrete = inference.get_concrete_function()
    frozen = convert_variables_to_constants_v2(concrete)
    graph_def = frozen.graph.as_graph_def()
    graph_ops = {node.op for node in graph_def.node}
    # Without unfolding recurrent loops, graph FLOPs are a LOWER BOUND.
    control_flow = bool(graph_ops & {'While', 'StatelessWhile', 'If', 'StatelessIf'})
    graph = tf.Graph()
    with graph.as_default():
        tf.compat.v1.import_graph_def(graph_def, name='')
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts['output'] = 'none'
        report = tf.compat.v1.profiler.profile(graph=graph, cmd='op', options=opts)
    if report is None or report.total_float_ops is None or report.total_float_ops <= 0:
        raise RuntimeError('TensorFlow graph profiler returned no usable operation count')
    return {'graph_profiled_flops_per_patch': int(report.total_float_ops),
            'graph_profiled_gflops_per_patch': float(report.total_float_ops)/1e9,
            'input_shape': [1, int(window_size), int(window_size), int(bands)],
            'method': 'tf.compat.v1.profiler float_operation, frozen inference graph, batch=1',
            'contains_control_flow': control_flow,
            'limitations': ('Control-flow bodies may be undercounted; graph profiling also '
                            'omits some non-MAC operations. Do not cite as exact FLOPs.'),
            'safe_to_claim_exact_flops': False}

# Train and Evaluate Model
def train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, HSID, teRatio, k, WS, DLM, RDHSI, GT,batch_size,epochs,output_dir):
    '''
    모델 명을 받아서 모델을 불러오고 입력된 데이터와 하이퍼파라미터를 바탕으로 모델을 학습하는 함수
    model_name: keras model : keras model
    Tr: np.array : train data
    TrC: np.array : train labels
    Va: np.array : validation data
    VaC: np.array : validation labels
    Te: np.array : test data
    TeC: np.array : test labels
    adam: keras optimizer : keras optimizer
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
    '''
    print(f'Model Name : {model_name}')
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(output_dir, "") + model_name.__name__
    os.makedirs(output_dir, exist_ok=True)
    print(model_name.__name__)

    # 모델 명이 FastKanMamba인 경우 dataset을 reshape
    if model_name == FastKanMamba:
        Tr = Tr.reshape(-1, WS,WS,k)  # Flatten input for the model
        Va = Va.reshape(-1, WS,WS,k)  # Flatten input for the model
        Te = Te.reshape(-1, WS,WS,k) 
        model = FastKanMamba(Tr, batch_size)
    else:
        model = model_name(WS,k,Num_Classes)
    
    # Model Summary
    file_name = f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_model_summary_of_{model_name.__name__}.txt"
    # Assuming `model` is your Keras model
    total_params = model.count_params()
    # Exact executed matrix-MAC counts for this fixed architecture. Not total FLOPs.
    mac_profile = estimate_instantiated_model_macs(model, WS, k)
    print(f"Analytical executed matrix MACs/patch: {mac_profile['macs_per_patch']:,}; "
          f"2-FLOP/MAC equivalent: {mac_profile['mac_equivalent_gflops_per_patch']:.6f} GFLOPs "
          '(excludes RBF/softmax/normalization).', flush=True)
    # Independent TensorFlow graph profile: may undercount control flow.
    try:
        flop_profile = profile_model_flops(model, WS, k)
        print(f"TF graph-profiled FLOPs/patch: "
              f"{flop_profile['graph_profiled_flops_per_patch']:,} "
              f"({flop_profile['graph_profiled_gflops_per_patch']:.6f} GFLOPs)")
        if flop_profile['contains_control_flow']:
            print('WARNING: recurrent/control-flow FLOPs may be undercounted; '
                  'do not cite the graph count as total model FLOPs.')
    except Exception as exc:
        flop_profile = {'graph_profiled_flops_per_patch': None,
                        'error': f'{type(exc).__name__}: {exc}',
                        'method': 'TF frozen-graph inference profiler; batch=1',
                        'safe_to_claim_exact_flops': False}
        warnings.warn(f'FLOPs profiling unavailable: {exc}')
    with open(os.path.join(output_dir, file_name),'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        fh.write(f"\nTotal Parameters: {total_params}\n")
        fh.write(f"Analytical matrix MAC profile: {json.dumps(mac_profile, indent=2)}\n")
        fh.write(f"FLOPs graph profiling: {json.dumps(flop_profile, indent=2)}\n")
    with open(os.path.join(output_dir, 'flops_profile.json'), 'w') as fh:
        json.dump({'model': model_name.__name__, 'parameters': total_params,
                   'analytical_matrix_macs': mac_profile,
                   'tf_graph_profile': flop_profile}, fh, indent=2)
    
    # Compiling the model (모델의 loss, optimizer, metrics 설정)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # Training the model
    start = time.time()
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]

    history = model.fit(x=Tr, y=TrC, batch_size=batch_size, epochs=epochs,
                        shuffle=True, validation_data=(Va, VaC), callbacks=callbacks)
    end = time.time()
    Tr_Time = end - start
    
    # Test Model
    start = time.time()
    Te_Pre = model.predict(Te)
    end = time.time()
    Te_Time = end - start
    
    ## Classification Report for Test Model
    classification,Confusion,OA,Per_Class,AA,Kappa = ClassificationReports(TeC, Te_Pre, target_names,zero_division=0)
    Te_Kappa = round(Kappa, 4)
    #Writing Results in CSV File
    file_name = os.path.join(output_dir,f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_{batch_size}_Classification_Report.csv")
    CSV_file = CSVResults(file_name, classification, Confusion, Tr_Time, Te_Time, DL_Time, Kappa, OA, AA, Per_Class)
    print(classification)
    # files.download(file_name)
     # T-SNE Plot
    # # Extract embeddings from the model (학습 데이터 임베딩 추출)
    # embeddings = model.get_embeddings(Tr)
    # # Convert embeddings to numpy array if it's not already (임베딩을 넘파이로 변환)
    # embeddings_np = embeddings.numpy()
    # # Apply t-SNE
    # tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    # embeddings_2d = tsne.fit_transform(embeddings_np)   # t-SNE 학습 및 변환
    # sns.set(style="whitegrid")
    # # Plot tSNE
    # plt.figure(figsize=(8, 5))
    # Optional embedding plot omitted; paper uses nipy_spectral exclusively
    # plt.title(f'tSNE: {HSID}', fontsize=12, fontweight='bold')
    # plt.xlabel('tSNE-1', fontsize=12, fontweight='bold')
    # plt.ylabel('tSNE-2', fontsize=12, fontweight='bold')
    # # Increase tick label size
    # plt.tick_params(labelsize=12)
    # # Add grid lines and set background color
    # plt.grid(True)
    # # Save the figure with a high DPI for better quality
    # plt.tight_layout()
    # file_name = f"{HSID}_{trRatio}_{k}_{WS}_tsne.png"
    # plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)
    
    # Paper figures are strictly based on held-out test predictions (not train/val).
    fig_dir = os.path.join(output_dir, "paper_figures")
    os.makedirs(fig_dir, exist_ok=True)
    pred_ids = np.argmax(Te_Pre, axis=1).astype(np.int32) + 1
    truth_ids = np.argmax(TeC, axis=1).astype(np.int32) + 1
    plot_learning_curves(history, fig_dir, HSID, WS, args.seed)
    plot_mac_breakdown(mac_profile, fig_dir, HSID, WS)
    with open(os.path.join(fig_dir, 'matrix_flops_components.csv'), 'w') as handle:
        handle.write('component,macs,2_flops_per_mac_equivalent\n')
        for component, macs in mac_profile['mac_breakdown'].items():
            handle.write(f'{component},{macs},{2*macs}\n')
    cm_test = plot_test_confusion(truth_ids, pred_ids, target_names,
                                  fig_dir, HSID, WS, args.split_seed)
    plot_class_recall(cm_test, target_names, fig_dir, HSID, WS, args.split_seed)
    with open(os.path.join(fig_dir, 'class_color_key.csv'), 'w', encoding='utf-8') as handle:
        handle.write('class_id,class_name,color_hex\n')
        for class_id, (class_name, rgb) in enumerate(
                zip(target_names, _hsi_class_palette(Num_Classes)), start=1):
            handle.write(f'{class_id},{json.dumps(class_name, ensure_ascii=False)},{to_hex(rgb)}\n')
    # Avoid an extra prediction pass for the scientifically valid test-only maps.
    if args.prediction_map in ("test", "full"):
        paper_maps = plot_heldout_classification_maps(
            GT, Te_xy, truth_ids, pred_ids, target_names,
            fig_dir, HSID, WS, args.split_seed)
        # Save pixel-aligned arrays for further visualization or exact audit.
        np.savez_compressed(os.path.join(fig_dir, "heldout_test_prediction_map.npz"),
                            true_test_map=paper_maps["test_ground_truth"],
                            predicted_test_map=paper_maps["test_predictions"],
                            correctness_map=paper_maps["correctness"],
                            test_coords=Te_xy, class_names=np.asarray(target_names))
    if args.prediction_map in ("test", "full"):
        # Match original GT_Plot visualization: predict all GT-labeled pixels.
        # This is NOT the held-out test map; test metrics above use Te_Pre only.
        print("Generating full predicted GT (visualization only; no test-score impact)...",
              flush=True)
        full_prediction = predict_labeled_map_in_batches(
            RDHSI, GT, model, WS, batch_size=max(256, batch_size))
        plot_full_predicted_gt_original_style(
            GT, full_prediction, fig_dir, HSID, WS, Num_Classes)
        print("Saved GT and predicted GT in original nipy_spectral style.", flush=True)
    with open(os.path.join(fig_dir, 'figure_protocol.json'), 'w') as handle:
        json.dump({
            'dataset': HSID, 'window_size': int(WS),
            'split_seed': int(args.split_seed), 'model_seed': int(args.seed),
            'test_count': int(len(truth_ids)),
            'test_correct_count': int(np.sum(truth_ids == pred_ids)),
            'test_predictions_only': True,
            'unplotted_test_pixel_value': 0,
            'class_labels_are_one_based': True,
            'classification_maps': ('test-only; non-test pixels masked'
                                    if args.prediction_map != 'none' else 'disabled'),
            'full_scene_if_present': 'visualization only; includes training, validation and excluded centers',
            'full_scene_figures_generated': bool(args.prediction_map in ('test', 'full')),

            'model_selection': 'last epoch (no early stopping, no best-weight restoration)'
        }, handle, indent=2)
    print('Saved publication figures under', fig_dir, flush=True)
    # One machine-readable record per seed; do not silently overwrite between runs.
    run_metrics = {
        "dataset": HSID, "window_size": int(WS),
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "visible_gpus": [g.name for g in tf.config.list_physical_devices('GPU')],
        "split_seed": int(args.split_seed), "model_seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "train_count": int(len(Tr)), "validation_count": int(len(Va)),
        "test_count": int(len(Te)),
        "original_labeled_centers": int(np.count_nonzero(GT)),
        "excluded_labeled_centers": int(np.count_nonzero(GT) - len(Tr) - len(Va) - len(Te)),
        "train_class_support": {str(c): int(n) for c, n in zip(*np.unique(GT[Tr_xy[:, 0], Tr_xy[:, 1]], return_counts=True))},
        "validation_class_support": {str(c): int(n) for c, n in zip(*np.unique(GT[Va_xy[:, 0], Va_xy[:, 1]], return_counts=True))},
        "test_class_support": {str(c): int(n) for c, n in zip(*np.unique(GT[Te_xy[:, 0], Te_xy[:, 1]], return_counts=True))},
        "OA": float(OA), "AA": float(AA), "Kappa": float(Kappa),
        "training_seconds": float(Tr_Time), "test_prediction_seconds": float(Te_Time),
        "parameters": int(total_params), "analytical_matrix_macs": mac_profile,
        "flops_profile": flop_profile,
        "max_epochs": int(epochs),
        "actual_epochs": int(len(history.history["loss"])),
        "best_validation_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "train_ratio_input": float(trRatio), "validation_ratio_input": float(vrRatio),
        "test_ratio_input": float(teRatio),
        "block_size": int(args.block_size), "buffer": int(args.spatial_buffer),
        "pca_bands": int(k), "pca_method": str(DLM),
        "batch_size": int(batch_size),
        "learning_rate": 0.001,
        "class_performance_pct": [float(v) for v in Per_Class],
        "source_version_note": "Original DiMA fixed-parameter affine recurrence (unchanged), with corrected token alignment; NOT selective Mamba or GRU",
        "target_token_flat_index": int(_target_token_index(WS, WS)),
        "architecture_version": "original_dima_affine_correct_token_alignment",
        "token_order": "row-major H*W, original spectral bands remain contiguous",
    }
    with open(os.path.join(output_dir, "results_summary.json"), "w") as f:
        json.dump(run_metrics, f, indent=2)
    print("Saved results_summary.json to", output_dir, flush=True)
    return history,Tr_Time, Te_Time, OA, AA,Kappa,total_params

# 

model_names = [FastKanMamba] 
all_results = {}
history_list = []
Tr_Time_list= []
Te_Time_list= []

for model_name in model_names:
    # Train and Evaluate Model
    history, Tr_Time, Te_Time, OA, AA,Kappa,total_params = train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC,
                                        adam, HSID, teRatio, k, WS,
                                        DLM, RDHSI, GT, batch_size,epochs,output_dir)
    history_list.append(history)
    Tr_Time_list.append(Tr_Time)
    Te_Time_list.append(Te_Time)
    
    # Save the results in a dictionary
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

# Define the filename for all results
all_results_filename = "all_models_results.json"

# Concatenate output_dir with the filename
all_results_output_path = os.path.join(output_dir, all_results_filename)

# Convert the dictionary to JSON format
json_data = json.dumps(all_results, indent=4)

# Optionally, write the JSON data to a file
with open(all_results_output_path, 'w') as json_file:
    json_file.write(json_data)

# Publication learning curves and other figures are generated inside
# train_and_evaluate_model using only the current run's actual arrays.
tf.keras.backend.clear_session()











# # Libraries
# import time
# import json
# import os 
# import argparse
# import random
# import sys
# # Select the GPU outside Python with CUDA_VISIBLE_DEVICES=0 (do not override it here).
# import numpy as np
# # Matplotlib (visualization)
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# # Sklearrn,Scipy, seaborn
# import scipy.io as sio  # MATLAB data file
# # Spectrak to Load HSI Dataset
# from operator import truediv
# # All spatial-split and reporting helpers live below in this SAME script.
# import warnings
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.metrics import (accuracy_score, classification_report,
#                              cohen_kappa_score, confusion_matrix)
# # Keras
# import keras
# from keras.losses import categorical_crossentropy
# from keras.models import Model, Sequential
# from keras.utils import to_categorical
# from keras.optimizers import legacy
# from keras import backend as Kb
# from keras.optimizers import Adam
# # TensorFlow
# import tensorflow as tf
# from tensorflow.keras import layers, Model
# from keras.layers import (Dense)
# from tensorflow.keras import regularizers
# from tensorflow.keras.layers import (Activation, Lambda, multiply)
# from tensorflow.keras.optimizers import Adam
# adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=1e-06)


# # import tensorflow as tf
# # print(tf.config.list_physical_devices('GPU'))
# # exit(0)
# # ============================================================================
# # SPATIALLY DISJOINT TRAIN / VALIDATION / TEST (INTEGRATED — NO EXTRA MODULE)
# # Ratio convention: train=.50 and val=.50 split the remaining 10% after test=.90.
# # Target = 5% train / 5% val / 90% test (actual retained ratios can vary).
# # A test/validation PATCH NEVER REUSES source pixels from another subset.
# # ============================================================================

# def _window_bounds(window_size, buffer=0):
#     """Exact source-pixel offsets for any even or odd window size."""
#     if window_size < 1 or buffer < 0:
#         raise ValueError("window_size must be >= 1, buffer must be >= 0")
#     before = window_size // 2 + buffer
#     after = window_size - 1 - window_size // 2 + buffer
#     return before, after


# def _contained_footprints(owner_map, subset_id, window_size, buffer):
#     """Return centers whose *entire* patch + safety buffer belongs to subset_id."""
#     height, width = owner_map.shape
#     before, after = _window_bounds(window_size, buffer)
#     side = before + after + 1
#     result = np.zeros((height, width), dtype=bool)
#     if height < side or width < side:
#         return result

#     binary = (owner_map == subset_id).astype(np.int32)
#     integral = np.pad(binary, ((1, 0), (1, 0)), mode="constant")
#     integral = integral.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)
#     sums = (integral[side:, side:] - integral[:-side, side:]
#             - integral[side:, :-side] + integral[:-side, :-side])
#     result[before:height - after, before:width - after] = sums == side * side
#     return result


# def resolve_split_ratios(train_ratio, validation_ratio, test_ratio):
#     """Resolve either absolute fractions or original two-stage split settings.

#     Absolute: (.70, .10, .20) -> (.70, .10, .20).
#     Two-stage: (.50, .50, .90) -> (.05, .05, .90): first hold out
#     90% test, then split remaining 10% equally between train and val.
#     """
#     values = np.asarray([train_ratio, validation_ratio, test_ratio], dtype=float)
#     if not np.all(np.isfinite(values)) or np.any(values <= 0) or np.any(values >= 1):
#         raise ValueError("All ratios must be finite and strictly between 0 and 1")
#     if np.isclose(values.sum(), 1.0, rtol=0, atol=1e-8):
#         return values / values.sum()
#     if np.isclose(train_ratio + validation_ratio, 1.0, rtol=0, atol=1e-8):
#         remaining = 1.0 - test_ratio
#         return np.array([remaining * train_ratio, remaining * validation_ratio,
#                          test_ratio], dtype=float)
#     raise ValueError("Ratios must either sum to 1 (absolute fractions), or "
#                      "train_ratio + validation_ratio must equal 1 "
#                      "(fractions of the non-test portion)")


# def _core_tile_histogram(gt, active, tile_of_labeled, classes_compact,
#                          block_size, window_size, buffer, num_tiles,
#                          num_classes):
#     """Conservative class counts always retained if a tile owns its interior."""
#     height, width = gt.shape
#     before, after = _window_bounds(window_size, buffer)
#     r, c = active[:, 0], active[:, 1]
#     tile_r, tile_c = r // block_size, c // block_size
#     r0, c0 = tile_r * block_size, tile_c * block_size
#     r1 = np.minimum(r0 + block_size, height)
#     c1 = np.minimum(c0 + block_size, width)
#     eligible = ((r - r0 >= before) & (c - c0 >= before) &
#                 (r + after < r1) & (c + after < c1))
#     hist = np.zeros((num_tiles, num_classes), dtype=np.int64)
#     np.add.at(hist, (tile_of_labeled[eligible], classes_compact[eligible]), 1)
#     return hist


# def _guided_owners(rng, core_hist, ratio):
#     """Prioritize class coverage in train then val before filling the test set."""
#     num_tiles, num_classes = core_hist.shape
#     owners = np.full(num_tiles, 2, dtype=np.int8)
#     core_count = core_hist.sum(axis=1)
#     available = core_count > 0
#     rarity = 1.0 / np.maximum(1, np.count_nonzero(core_hist > 0, axis=0))
#     # Cover training classes first, then validation classes if possible.
#     for group in (0, 1):
#         missing = np.any(core_hist > 0, axis=0)
#         while missing.any() and available.any():
#             gains = (core_hist[:, missing] > 0) @ rarity[missing]
#             gain_score = gains / np.maximum(core_count, 1)**0.25
#             gain_score *= rng.uniform(0.90, 1.10, size=num_tiles)
#             gain_score[~available] = -1
#             chosen = int(np.argmax(gain_score))
#             if gains[chosen] <= 0:
#                 break
#             owners[chosen] = group
#             available[chosen] = False
#             missing &= core_hist[chosen] == 0
#     # Complete the small train/validation quotas without splitting a tile.
#     global_total = int(core_count.sum())
#     for group in (0, 1):
#         target = ratio[group] * global_total
#         have = int(core_count[owners == group].sum())
#         while have < target and available.any():
#             candidates = np.flatnonzero(available)
#             cost = np.abs(have + core_count[candidates] - target)
#             cost = cost + rng.uniform(0, 0.02, size=len(cost)) * np.maximum(1, target)
#             idx = int(candidates[np.argmin(cost)])
#             # Do not add a block that moves us farther away than doing nothing.
#             if abs(have + core_count[idx] - target) >= abs(have - target):
#                 break
#             owners[idx] = group
#             available[idx] = False
#             have += core_count[idx]
#     return owners


# def disjoint_spatial_split(
#     gt, window_size, train_ratio=0.70, validation_ratio=0.10,
#     test_ratio=0.20, block_size=32, buffer=0, seed=345,
#     search_trials=512,
# ):
#     """Partition labeled HSI centers so train/val/test patches share ZERO pixels.

#     Ratios may be absolute (sum to 1) or legacy conditional
#     train/validation fractions of the non-test portion. They are targets,
#     not guarantees, after block quantization and spatial boundary erosion.
#     Returns (train_coords, val_coords, test_coords, split_map, owner_map).
#     """
#     gt = np.asarray(gt)
#     if gt.ndim != 2 or not np.issubdtype(gt.dtype, np.integer):
#         raise ValueError("gt must be a 2-D array of integer class IDs (0=unlabeled)")
#     if np.any(gt < 0):
#         raise ValueError("gt must contain only nonnegative class IDs")
#     ratios = resolve_split_ratios(train_ratio, validation_ratio, test_ratio)
#     if not isinstance(block_size, (int, np.integer)) or block_size < window_size + 2 * buffer:
#         raise ValueError("block_size must be an integer >= window_size + 2 * buffer")
#     if search_trials < 1:
#         raise ValueError("search_trials must be >= 1")

#     height, width = gt.shape
#     rows = (height + block_size - 1) // block_size
#     cols = (width + block_size - 1) // block_size
#     num_tiles = rows * cols
#     active = np.argwhere(gt > 0)
#     if len(active) == 0:
#         raise ValueError("gt has no labeled pixels")
#     tile_of_labeled = ((active[:, 0] // block_size) * cols
#                        + (active[:, 1] // block_size))
#     class_labels = np.unique(gt[gt > 0])
#     classes_compact = np.searchsorted(class_labels, gt[active[:, 0], active[:, 1]])
#     hist = np.zeros((num_tiles, len(class_labels)), dtype=np.int64)
#     np.add.at(hist, (tile_of_labeled, classes_compact), 1)
#     active_tiles = np.flatnonzero(hist.sum(axis=1))
#     if len(active_tiles) < 3:
#         raise ValueError(f"Only {len(active_tiles)} labeled spatial blocks: need >= 3. "
#                          "Decrease --block_size if the scene has enough spatial extent")
#     core_hist = _core_tile_histogram(gt, active, tile_of_labeled,
#                                      classes_compact, block_size, window_size,
#                                      buffer, num_tiles, len(class_labels))
#     if not np.any(core_hist):
#         raise ValueError("No labeled pixels survive inside individual spatial blocks. "
#                          "Reduce window size/buffer or adjust block_size")
#     # Candidate generation uses core-tile counts (guaranteed retained pixels).
#     # Final selection rescans actual post-erosion footprints, not raw tile counts.
#     rng = np.random.default_rng(seed)
#     core_totals = core_hist.sum(axis=0)
#     all_total = int(core_totals.sum())
#     potential = np.count_nonzero(core_hist > 0, axis=0) >= 3
#     # Small owner regions lose a larger fraction of centers to erosion than
#     # the much larger contiguous test region. Explore several OVER-ALLOCATIONS
#     # of blocks to small splits, then choose on exact post-erosion ratios.
#     scales = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25) if ratios[2] >= .65 else (1.0,)
#     candidates_by_scale = [[] for _ in scales]
#     for trial in range(search_trials):
#         scale_id = trial % len(scales)
#         scale = scales[scale_id]
#         proposal = ratios.copy()
#         proposal[:2] *= scale
#         proposal[2] = 1.0 - proposal[0] - proposal[1]
#         if proposal[2] <= 0:
#             continue
#         if trial % 4 != 0:
#             owners = _guided_owners(rng, core_hist, proposal)
#         else:
#             owners = rng.choice(3, size=num_tiles, p=proposal).astype(np.int8)
#         counts = np.stack([core_hist[owners == j].sum(axis=0) for j in range(3)])
#         totals = counts.sum(axis=1)
#         if (totals == 0).any():
#             continue
#         actual = totals / totals.sum()
#         per_cls = counts / np.maximum(1, core_totals)[None, :]
#         score = (60.0 * np.sum((actual - proposal)**2)
#                  + 0.5 * np.mean((per_cls - proposal[:, None])**2)
#                  + 1_000.0 * np.sum(counts[0, core_totals > 0] == 0)
#                  + 0.2 * np.sum(counts[1:, potential] == 0))
#         candidates_by_scale[scale_id].append((float(score), owners))
#     candidates = []
#     for bucket in candidates_by_scale:
#         bucket.sort(key=lambda item: item[0])
#         candidates.extend(bucket[:8])
#     if not candidates:
#         raise ValueError("Unable to allocate three nonempty spatial partitions; "
#                          "try a smaller block size or different seed")
#     # Score finalists on *eligible centers* after eroding partition boundaries.
#     tile_ids = ((np.arange(height)[:, None] // block_size) * cols
#                 + np.arange(width)[None, :] // block_size)
#     best = None
#     best_score = np.inf
#     for _proxy_score, owners in candidates:
#         owner_map = owners[tile_ids]
#         masks = []
#         counts = []
#         for subset_id in range(3):
#             mask = _contained_footprints(owner_map, subset_id, window_size, buffer)
#             mask &= gt > 0
#             masks.append(mask)
#             counts.append(np.bincount(np.searchsorted(class_labels, gt[mask]),
#                                       minlength=len(class_labels)))
#         totals = np.array([c.sum() for c in counts])
#         if (totals == 0).any() or np.any(counts[0] == 0):
#             continue
#         kept_ratios = totals / totals.sum()
#         class_props = np.stack(counts) / np.maximum(1, np.stack(counts).sum(axis=0))[None, :]
#         score = (60.0 * np.sum((kept_ratios - ratios)**2)
#                  + np.mean((class_props - ratios[:, None])**2)
#                  + 0.2 * np.sum(np.stack(counts)[1:, potential] == 0))
#         if score < best_score:
#             best_score = score
#             best = (owner_map.copy(), masks)
#     if best is None:
#         raise ValueError(
#             "No candidate retained all GT classes in training with nonempty validation "
#             "and test regions AFTER spatial erosion. For a 5/5/90 split, rare classes "
#             "may require >5% training. Reduce --block_size or --spatial_buffer, "
#             "increase --search_trials, or try another --split_seed. "
#             "Do not silently drop training classes."
#         )
#     owner_map, masks = best
#     split_map = np.zeros((height, width), dtype=np.uint8)
#     coords = []
#     for i, mask in enumerate(masks):
#         split_map[mask] = i + 1
#         coords.append(np.argwhere(mask).astype(np.int32))
#     for name, part in [("validation", coords[1]), ("testing", coords[2])]:
#         present = np.unique(gt[part[:, 0], part[:, 1]])
#         missing = np.setdiff1d(class_labels, present)
#         if len(missing):
#             warnings.warn(f"{name} partition has no pixels for classes "
#                           f"{missing.tolist()}; class-specific evaluation is undefined",
#                           stacklevel=2)
#     verify_disjoint_footprints(coords, gt.shape, window_size, owner_map)
#     return (*coords, split_map, owner_map)


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


# def predict_labeled_map_in_batches(hsi, gt, model, window_size, batch_size=256):
#     """Predict the full labeled scene for visualization with constant-size batches."""
#     before, after = _window_bounds(window_size)
#     padded = np.pad(hsi, ((before, after), (before, after), (0, 0)))
#     coords = np.argwhere(gt > 0)
#     output = np.zeros(gt.shape, dtype=np.int32)
#     steps = np.arange(window_size)
#     for start in range(0, len(coords), batch_size):
#         chunk = coords[start:start + batch_size]
#         rr = chunk[:, 0, None, None] + steps[None, :, None]
#         cc = chunk[:, 1, None, None] + steps[None, None, :]
#         patches = padded[rr, cc].astype(np.float32, copy=False)
#         prediction = model.predict(patches, verbose=0)
#         output[chunk[:, 0], chunk[:, 1]] = prediction.argmax(axis=1) + 1
#     return output


# # ============================================================================
# # CLASSIFICATION AND CSV REPORTING (FORMERLY IMPORTED FROM utils.py)
# # ============================================================================

# def ClassificationReports(TeC, Te_Pred, target_names,zero_division=0):
#     '''
#     최종 분류 및 혼동 행렬을 위한 클래스 레이블 할당 함수
#     TeC: np.array : test data
#     Te_Pred: np.array : test prediction
#     target_names: list : target names
#     zero_division: int : zero division
    
#     return
#     classification: str : classification report
#     confusion: np.array : confusion matrix
#     oa: float : overall accuracy
#     each_acc: np.array : each accuracy
#     aa: float : average accuracy
#     kappa: float : kappa coefficient
#     '''
#     Te_Pred = np.argmax(Te_Pred, axis=1)    # Get the argmax of the test prediction
#     # scikit-learn classification report 생성 (precision, recall, F1-score etc.)
#     classification = classification_report(np.argmax(TeC, axis=1), Te_Pred,
#                                            labels=np.arange(len(target_names)),
#                                            target_names=target_names,
#                                            zero_division=zero_division)
#     oa = accuracy_score(np.argmax(TeC, axis=1), Te_Pred)    # Overall accuracy, OA
#     confusion = confusion_matrix(np.argmax(TeC, axis=1), Te_Pred,
#                                  labels=np.arange(len(target_names)))   # Confusion matrix
#     list_diag = np.diag(confusion) # Get the diagonal of the confusion matrix (각 클래스별 정분류 개수)
#     list_raw_sum = np.sum(confusion, axis=1) # Get the sum of the rows of the confusion matrix (각 클래스별 실제 샘플 개수)
#     each_acc = np.divide(list_diag, list_raw_sum,
#                          out=np.full(len(list_diag), np.nan, dtype=float),
#                          where=list_raw_sum != 0)
#     aa = float(np.nanmean(each_acc))  # AA only over classes with test support
#     if np.any(list_raw_sum == 0):
#         warnings.warn("Some classes have zero test support; their accuracy is undefined and excluded from AA")
#     kappa = cohen_kappa_score(np.argmax(TeC, axis=1), Te_Pred)  # Kappa coefficient
#     return classification, confusion, oa*100, each_acc*100, aa*100, kappa*100


# def CSVResults(file_name, classification, confusion, Tr_Time, Te_Time, DL_Time, kappa, oa, aa, each_acc, memory_used=None):
#     '''
#     결과 csv 파일에 쓰기
#     file_name: str : file name
#     classification: str : classification report
#     confusion: np.array : confusion matrix
#     Tr_Time: float : training time
#     Te_Time: float : test time
#     DL_Time: float : dimensionality reduction time
#     kappa: float : kappa coefficient
#     oa: float : overall accuracy
#     aa: float : average accuracy
#     each_acc: np.array : each accuracy
#     '''
#     classification = str(classification)
#     confusion = str(confusion)
#     with open(file_name, 'w') as CSV_file:
#       CSV_file.write('{} Tr_Time'.format(Tr_Time))
#       CSV_file.write('\n')
#       CSV_file.write('{} Te_Time'.format(Te_Time))
#       CSV_file.write('\n')
#       CSV_file.write('{} DL_Time'.format(DL_Time))
#       CSV_file.write('\n')
#       CSV_file.write('{} Kappa accuracy (%)'.format(kappa))
#       CSV_file.write('\n')
#       CSV_file.write('{} Overall accuracy (%)'.format(oa))
#       CSV_file.write('\n')
#       CSV_file.write('{} Average accuracy (%)'.format(aa))
#       CSV_file.write('\n')
#       if memory_used is not None:
#             CSV_file.write('{} Inference Memory Usage (MB)\n'.format(round(memory_used, 2)))
#       CSV_file.write('\n')
#       CSV_file.write('{}'.format(classification))
#       CSV_file.write('\n')
#       CSV_file.write('{}'.format(each_acc))
#       CSV_file.write('\n')
#       CSV_file.write('{}'.format(confusion))
#     return CSV_file

# #Create an ArgumentParser object
# parser = argparse.ArgumentParser(description="Script Description")
# # Add arguments for each parameter
# parser.add_argument(
#     "--dataset",
#     "-d",
#     type=str,
#     choices=["LK", "HC", "UH13", "PU", "PC", "SA","HH","QUH","PI","TH"],
#     default="PI",
#     help="Dataset identifier",
# )
# parser.add_argument("--window_size", "-ws", type=int, default=4, help="Window size")    
# parser.add_argument("--dl_method", "-dlm", type=str, default='PCA', help="Dimension Reduction Method")  # 차원 축소 메소드
# parser.add_argument("--test_ratio", "-te", type=float, default=0.90, help="Test holdout fraction; accepts legacy 0.50/0.50/0.90")
# parser.add_argument("--train_ratio", "-tr", type=float, default=0.50, help="Training fraction of remaining non-test data when train+val=1")
# parser.add_argument(
#     "--validation_ratio", "-vr", type=float, default=0.50, help="Validation fraction of remaining non-test data when train+val=1"
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
# parser.add_argument("--block_size", type=int, default=32,
#                     help="Side length of assigned non-overlapping spatial blocks")
# parser.add_argument("--spatial_buffer", type=int, default=0,
#                     help="Extra source pixels to exclude around each partition boundary")
# parser.add_argument("--search_trials", type=int, default=512,
#                     help="Random/guided block allocation trials")
# parser.add_argument("--seed", type=int, default=345,
#                     help="Run seed: random, NumPy, TensorFlow initialization and training; also split unless overridden")
# parser.add_argument("--split_seed", type=int, default=None,
#                     help="Optional fixed spatial split seed (defaults to --seed)")
# parser.add_argument("--deterministic", action="store_true",
#                     help="Request TensorFlow deterministic GPU operations (may reduce performance)")
# parser.add_argument("--prediction_map", choices=["none", "test", "full"], default="none",
#                     help="Map mode; 'test' reuses predictions, 'full' is expensive")
# parser.add_argument("--verify_split_only", action="store_true",
#                     help="Verify the spatial split and exit without PCA or model training")
# # Parse the command-line arguments
# args = parser.parse_args()
# # Set before importing TensorFlow. Python's hash seed must be exported before
# # interpreter startup for full hash reproducibility (e.g. PYTHONHASHSEED=42).
# # Seed model initialization, dropout, and fit shuffling before any layer is created.
# tf.keras.utils.set_random_seed(args.seed)
# if args.deterministic:
#     tf.config.experimental.enable_op_determinism()
# if args.deterministic:
#     os.environ["TF_DETERMINISTIC_OPS"] = "1"
# if args.split_seed is None:
#     args.split_seed = args.seed
# random.seed(args.seed)
# np.random.seed(args.seed)
# print(f"Seeds: split={args.split_seed} model={args.seed}; deterministic={args.deterministic}")

# # DataSet Loader Funtion
# base_dir = './hsi_datasets'
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
#     else:
#         print("Wrong Choice")

#     return HSI, GT, Num_Classes, target_names

# # Global Parameters
# HSID = args.dataset ## "SLA", "IP", "PU", "PC", "SA", "KSC", "BS", "LK", "HH" (difficult to compile), "HC"
# DLM = args.dl_method ## "PCA", "iPCA"
# WS = args.window_size
# # Using the calculated values
# trRatio = args.train_ratio
# vrRatio = args.validation_ratio
# teRatio = args.test_ratio
# effective_ratios = resolve_split_ratios(trRatio, vrRatio, teRatio)
# print("Requested ratios (train, val, test):", trRatio, vrRatio, teRatio)
# print("Effective TARGET fractions (train, val, test):", effective_ratios.tolist())
# k = args.number_of_bands
# num_heads=8
# out_channels=32
# epochs = args.epochs
# batch_size = args.batch_size
# # output_dir = os.path.join(f"Spaital_Spectral_Kan_Mamba/{HSID}/window_size/{WS}")
# output_dir = os.path.join(
#     f"KanMamba_{num_heads}_chnal_{out_channels}_bs_{batch_size}",
#     HSID, str(WS), "arch_original_dima_ssm_tokenfix", f"split_{args.split_seed}",
#     f"seed_{args.seed}")
# if not os.path.exists(output_dir):
#       os.makedirs(output_dir)

# # %%
# # Spatially disjoint patch extraction replaces the old ImageCubes function.
# HSI, GT, Num_Classes, target_names = LoadHSIData(HSID)  # Load the HSI data
# # STEP 1: Spatially assign blocks on the ORIGINAL image, before PCA or patches.
# # Block erosion removes all patch centers whose input footprints cross splits.
# Tr_xy, Va_xy, Te_xy, split_map, owner_map = disjoint_spatial_split(
#     GT, window_size=WS,
#     train_ratio=trRatio, validation_ratio=vrRatio, test_ratio=teRatio,
#     block_size=args.block_size, buffer=args.spatial_buffer, seed=args.split_seed,
#     search_trials=args.search_trials,
# )

# # Save the exact split so it can be reproduced and independently audited.
# np.savez_compressed(
#     os.path.join(output_dir, "disjoint_spatial_split.npz"),
#     train_coords=Tr_xy, validation_coords=Va_xy, test_coords=Te_xy,
#     split_map=split_map, owner_map=owner_map,
#     window_size=WS, block_size=args.block_size,
#     spatial_buffer=args.spatial_buffer, seed=args.split_seed, model_seed=args.seed,
#     requested_ratios=np.array([trRatio, vrRatio, teRatio]),
#     effective_ratios=effective_ratios,
# )

# n_selected = len(Tr_xy) + len(Va_xy) + len(Te_xy)
# n_labeled = int(np.count_nonzero(GT))
# print(f"Original labeled centers: {n_labeled:,}")
# for name, coords in (("Train", Tr_xy), ("Validation", Va_xy), ("Test", Te_xy)):
#     print(f"{name}: {len(coords):,} samples ({100 * len(coords)/n_selected:.2f}% of retained)")
# print(f"Excluded by spatial disjointness: {n_labeled - n_selected:,} labeled centers")
# if n_selected:
#     kept = np.array([len(Tr_xy), len(Va_xy), len(Te_xy)]) / n_selected
#     print("Achieved fractions AFTER erosion:", np.round(kept, 4).tolist())
#     if np.any(np.abs(kept - effective_ratios) > 0.025):
#         print("WARNING: retained ratios differ by >2.5 percentage points from targets; "
#               "spatially disjoint blocks cannot promise exact split percentages.")
# for group_name, coords in (("train", Tr_xy), ("validation", Va_xy), ("test", Te_xy)):
#     labels, counts = np.unique(GT[coords[:, 0], coords[:, 1]], return_counts=True)
#     print(group_name, "class support:", dict(zip(labels.tolist(), counts.tolist())))
#     sparse = {int(label): int(count) for label, count in zip(labels, counts) if count < 20}
#     if sparse and group_name != "test":
#         warnings.warn(f"{group_name} has fewer than 20 centers for classes {sparse}; "
#                       "pre-register how sparse classes are handled before publication")
# # verify_disjoint_footprints is also called internally by disjoint_spatial_split.

# fig, ax = plt.subplots(figsize=(7, 7))
# ax.imshow(split_map, cmap=ListedColormap(["#222222", "#4082c4", "#f2af32", "#d94c4c"]),
#           interpolation="nearest", vmin=0, vmax=3)
# ax.set_title("Spatially disjoint pixel-center partitions: train / val / test")
# ax.axis("off")
# fig.savefig(os.path.join(output_dir, "disjoint_split_map.png"), dpi=250, bbox_inches="tight")
# plt.close(fig)

# if args.verify_split_only:
#     print("Split-only verification completed. No PCA or network training performed.")
#     raise SystemExit(0)

# # STEP 2: PCA is fitted ONLY on training-center pixels (inductive evaluation).
# start = time.time()
# RDHSI, pca = reduce_pca_training_only(HSI, Tr_xy, n_components=k, method=DLM)
# DL_Time = time.time() - start

# # STEP 3: Extract only the three selected, non-overlapping patch subsets.
# Tr, TrC = extract_patches_at_coords(RDHSI, GT, Tr_xy, window_size=WS)
# Va, VaC = extract_patches_at_coords(RDHSI, GT, Va_xy, window_size=WS)
# Te, TeC = extract_patches_at_coords(RDHSI, GT, Te_xy, window_size=WS)


# TrC = to_categorical(TrC, num_classes=Num_Classes)
# VaC = to_categorical(VaC, num_classes=Num_Classes)
# TeC = to_categorical(TeC, num_classes=Num_Classes)
# print("Test inputs:", Te.shape, "Test targets:", TeC.shape)

# # Token Generation
# # Patch coordinates follow the SAME convention as extract_patches_at_coords():
# # target pixel is at local (H // 2, W // 2), including even-sized windows.
# def _target_token_index(height, width):
#     """Zero-based row-major index of the labeled target pixel in an HxW patch."""
#     return (height // 2) * width + (width // 2)


# def _flatten_hsi_tokens(x):
#     """Convert (batch, row, column, bands) to row-major (batch, H*W, bands).

#     Preserve all spectral bands together for each spatial location. In particular,
#     never transpose spatial and spectral dimensions before flattening.
#     """
#     shape = tf.shape(x)
#     return tf.reshape(x, [shape[0], shape[1] * shape[2], shape[3]])


# class SpectralSpatialTokenGeneration(tf.keras.layers.Layer):
#     """Two independently learned embeddings of spatially aligned pixel spectra.

#     Both branches use the SAME row-major pixel order, so token i in each branch
#     corresponds to the same input pixel at fusion. Subsequent attention models
#     spatial context. No separate spatial convolution is claimed here.
#     """
#     def __init__(self, out_channels, **kwargs):
#         super().__init__(**kwargs)
#         self.spatial_tokens = Dense(out_channels)
#         self.spectral_tokens = Dense(out_channels)

#     def call(self, x):
#         pixel_spectra = _flatten_hsi_tokens(x)
#         return self.spatial_tokens(pixel_spectra), self.spectral_tokens(pixel_spectra)


# # Fast Kolomgorov Arnold Network
# class SplineLinear(tf.keras.layers.Layer):
#     def __init__(self, in_features, out_features, init_scale=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.in_features = in_features
#         self.out_features = out_features
#         self.init_scale = init_scale
#         self.weight = self.add_weight(
#             shape=(self.out_features, self.in_features),
#             initializer=tf.keras.initializers.GlorotUniform(), #tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=init_scale), #HeNormal(),
#             trainable=True,
#             regularizer=tf.keras.regularizers.l2(0.001)
#         )
#     def call(self, inputs):
#         return tf.matmul(inputs, tf.transpose(self.weight))
    
    
# # Radial Basis Function for Fast KAN
# class RadialBasisFunction(tf.keras.layers.Layer):
#     def __init__(self, grid_min=-1.0, grid_max=1.0, num_grids=8, denominator=None, **kwargs):
#         super().__init__(**kwargs)
#         self.grid_min = grid_min
#         self.grid_max = grid_max
#         self.num_grids = num_grids
#         self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
#         self.grid = tf.Variable(
#             initial_value=tf.linspace(grid_min, grid_max, num_grids),
#             trainable=True,
#             name="grid"
#         )
#     def call(self, inputs):
#         expanded_inputs = tf.expand_dims(inputs, axis=-1)
#         return tf.exp(-((expanded_inputs - self.grid) / self.denominator) ** 2)


# # Fast Kolmogorov Arnold Network Layer
# class FastKANLayer(tf.keras.layers.Layer):
#     def __init__(self, input_dim, output_dim, grid_min=-1.0, grid_max=1.0, 
#                  num_grids=8, use_base_update=True, use_layernorm=True, base_activation=tf.nn.swish,spline_weight_init_scale=0.1, 
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.use_layernorm = use_layernorm
#         self.use_base_update = use_base_update
#         self.base_activation = base_activation
#         if use_layernorm:
#             assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
#             self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)
#         else:
#             self.layernorm = None
#         self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
#         self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, init_scale=spline_weight_init_scale)
#         if use_base_update:
#             self.base_activation = base_activation
#             self.base_linear = tf.keras.layers.Dense(output_dim)

#     def call(self, inputs, use_layernorm=True):
#         if self.layernorm is not None and use_layernorm:
#             normalized_inputs = self.layernorm(inputs)
#         else:
#             normalized_inputs = inputs
#         spline_basis = self.rbf(normalized_inputs)
#         spline_basis_flat = tf.reshape(
#             spline_basis,
#             shape=(-1, spline_basis.shape[-2] * spline_basis.shape[-1])
#         )
#         ret = self.spline_linear(spline_basis_flat)
#         # ret = tf.reshape(ret, shape=tf.shape(inputs)) 
#         output_shape = tf.concat([tf.shape(inputs)[:-1], [self.output_dim]], axis=0)
#         ret = tf.reshape(ret, shape=output_shape)
#         if self.use_base_update:
#             base = self.base_linear(self.base_activation(inputs))
#             ret = ret + base
#         return ret


# # 멀티 헤드 어텐션을 Fast KAN으로 구현
# # Multihead Attention With Fast Kolmogorov Arnold Network
# class MultiHeadKANAttention(tf.keras.layers.Layer):
#     def __init__(self, 
#                  input_dim, num_heads, output_dim, grid_min=-1.0, grid_max=1.0, num_grids=8, use_base_update=False,base_activation=tf.nn.swish, spline_weight_init_scale=0.1, dropout_rate=0.7, 
#                  output_activation=None,  # Optional activation for output
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.num_heads = num_heads
#         self.head_dim = output_dim // num_heads
#         self.output_dim = output_dim
#         self.all_head_size = self.num_heads * self.head_dim
        
#         # Query, Key, and Value projections via FastKANLayer
#         self.query_kan = FastKANLayer(input_dim=input_dim,output_dim=self.all_head_size,grid_min=grid_min,grid_max=grid_max, 
#             num_grids=num_grids,use_base_update=use_base_update,base_activation=base_activation,
#             spline_weight_init_scale=spline_weight_init_scale
#         )
#         self.key_kan = FastKANLayer(input_dim=input_dim,output_dim=self.all_head_size,grid_min=grid_min,grid_max=grid_max, 
#             num_grids=num_grids,use_base_update=use_base_update,base_activation=base_activation,
#             spline_weight_init_scale=spline_weight_init_scale
#         )
#         self.value_kan = FastKANLayer(input_dim=input_dim,output_dim=self.all_head_size,grid_min=grid_min,grid_max=grid_max, 
#             num_grids=num_grids,use_base_update=use_base_update,base_activation=base_activation,
#             spline_weight_init_scale=spline_weight_init_scale
#         )
#         # Final output projection
#         self.output_projection = tf.keras.layers.Dense(units=self.output_dim) # Better weight initialization
#         # Optional output activation
#         self.output_activation = tf.keras.layers.Activation(output_activation) if output_activation else None
#         # Dropout for attention weights
#         self.attention_dropout = tf.keras.layers.Dropout(dropout_rate)

#     def split_heads(self, x, batch_size):
#         """
#         Splits the last dimension of the input into (num_heads, head_dim)
#         Transpose the resulting tensor to (batch_size, num_heads, seq_len, head_dim)
#         """
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
#         return tf.transpose(x, perm=[0, 2, 1, 3])

#     def combine_heads(self, x):
#         """
#         Combines the (num_heads, head_dim) into a single dimension.
#         Reshapes tensor to (batch_size, seq_len, output_dim).
#         """
#         x = tf.transpose(x, perm=[0, 2, 1, 3])
#         return tf.reshape(x, (tf.shape(x)[0], -1, self.num_heads * self.head_dim))

#     def call(self, queries, keys, values, training=False):
#         batch_size = tf.shape(queries)[0]
#         # Compute Query, Key, and Value embeddings using FastKANLayer
#         Q = self.query_kan(queries)
#         K = self.key_kan(keys)
#         V = self.value_kan(values)

#         # Split heads into (batch_size, num_heads, seq_len, head_dim)
#         Q = self.split_heads(Q, batch_size)
#         K = self.split_heads(K, batch_size)
#         V = self.split_heads(V, batch_size)

#         # Scaled dot-product attention
#         attention_scores = tf.matmul(Q, K, transpose_b=True)  # Shape: (batch_size, num_heads, seq_len, seq_len)
#         d_k = tf.cast(self.head_dim, tf.float32)  # Head dimension
#         attention_scores /= tf.math.sqrt(d_k)  # Scale by sqrt(head_dim) for numerical stability
#         attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # Softmax over sequence length
#         # Dropout on attention weights
#         attention_weights = self.attention_dropout(attention_weights, training=training)
#         # Weighted sum of Value embeddings
#         attention_output = tf.matmul(attention_weights, V)  # Shape: (batch_size, num_heads, seq_len, head_dim)
#         # Combine heads back to (batch_size, seq_len, output_dim)
#         attention_output = self.combine_heads(attention_output)
#         # Final linear projection
#         output = self.output_projection(attention_output)

#         # Optional activation function
#         if self.output_activation:
#             output = self.output_activation(output)
#         return output
    
# class SpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
#     def __init__(self, out_channels, **kwargs):
#         super(SpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
        
#         self.spatial_gate = Sequential([
#             FastKANLayer(input_dim=out_channels,output_dim=out_channels,
#             grid_min=-1.0,grid_max=1.0,num_grids=8) ,
#             Activation('sigmoid'),
#         ])

#         self.spectral_gate = Sequential([
#             # FastKANLayer(input_dim=out_channels, output_dim=out_channels),
#             FastKANLayer(input_dim=out_channels,output_dim=out_channels,
#              grid_min=-1.0,grid_max=1.0,num_grids=8),
#             Activation('sigmoid'),
#         ])

#     def call(self, spatial_tokens, spectral_tokens, center_tokens):
#         spatial_weights = self.spatial_gate(center_tokens)
#         spectral_weights = self.spectral_gate(center_tokens)
#         spatial_enhanced = spatial_tokens * tf.expand_dims(spatial_weights, axis=1)
#         spectral_enhanced = spectral_tokens * tf.expand_dims(spectral_weights, axis=1)
#         return spatial_enhanced, spectral_enhanced

# # Original fixed-parameter affine StateSpaceModel from DiMA, unmodified.
# class StateSpaceModel(tf.keras.layers.Layer):
#     def __init__(self, state_dim, **kwargs):
#         super(StateSpaceModel, self).__init__(**kwargs)
#         self.state_dim = state_dim
#         self.state_transition = Dense(state_dim)
#         self.state_update = Dense(state_dim)
#     def call(self, x):
#         state = tf.zeros([tf.shape(x)[0], self.state_dim])
#         for t in range(tf.shape(x)[1]):
#             state = self.state_transition(state) + self.state_update(x[:, t, :])
#         return state
# # 위에 나온 것들 모두 사용한 전체 모델
# # Architecture
# class SSFKSMambaModel(tf.keras.Model):
#     '''
#     token_generation -> feature_enhancement -> multi_head_attention -> concat -> state_space_model -> classifier(Dense)
#     HSI 데이터를 받아서 token_generation을 통해 공간과 스펙트럴 토큰을 생성하고 
#     spatial_tokens의 중간 토큰을 gating으로 사용하여 
#     feature_enhancement를 통해 토큰을 강화하고
#     공간과 스펙트럴 토큰 각각에 multi_head_attention을 적용하고
#     어텐션 결과를 concat하여 state_space_model을 통해 시계열 모델링을 하고
#     ssm의 결과를 classifier(Dense)에 통과시켜서 분류
#     '''
#     def __init__(self, out_channels, state_dim=64, num_heads=num_heads, num_grids=8, **kwargs):
#         super(SSFKSMambaModel, self).__init__(**kwargs)
#         # Tokenization
#         self.token_generation = SpectralSpatialTokenGeneration(out_channels)
#         # Multi-Head Attention
#         self.multi_head_attention = MultiHeadKANAttention(input_dim=out_channels, output_dim=out_channels, num_heads=num_heads, grid_min=-1.0,
#             grid_max=1.0,num_grids=num_grids,base_activation=tf.nn.swish)
#         # Feature Enhancement
#         self.feature_enhancement = SpectralSpatialFeatureEnhancement(out_channels)
#         # Original unmodified fixed-parameter affine recurrent aggregation from DiMA
#         self.state_space_model = StateSpaceModel(state_dim=state_dim)
#         # Dense Layers and Classification
#         self.dense = tf.keras.layers.Dense(units=out_channels, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))
#         self.dropout = tf.keras.layers.Dropout(0.5)
#         self.classifier = tf.keras.layers.Dense(Num_Classes, activation='softmax')

#     # Forward pass
#     def call(self, x, training=False):
#         # Step 1: Tokenization
#         spatial_tokens, spectral_tokens = self.token_generation(x)
#         # Even WS=12 has a target at (6,6): row-major index 6*12+6 = 78.
#         # Use runtime dimensions for dynamic batch/image shapes.
#         target_index = _target_token_index(tf.shape(x)[1], tf.shape(x)[2])
#         center_tokens = tf.gather(spatial_tokens, target_index, axis=1)
#         # Step 2: Feature Enhancement
#         spatial_enhanced, spectral_enhanced = self.feature_enhancement(spatial_tokens, spectral_tokens, center_tokens)
#         # Step 3: Apply Multi-Head Attention
#         spatial_attention = self.multi_head_attention(spatial_enhanced,spatial_enhanced,spatial_enhanced)
#         spectral_attention = self.multi_head_attention(spectral_enhanced,spectral_enhanced,spectral_enhanced)
#         # Step 4: Combine the outputs
#         combined_output = tf.concat([spatial_attention, spectral_attention], axis=-1)
#         # Step 5: Original DiMA fixed-parameter affine recurrent aggregation
#         state_output = self.state_space_model(combined_output)
#         # Step 6: Dense layer and classification
#         output = self.classifier(state_output)
#         return output

# # 모델 생성 함수
# def FastKanMamba(Tr, batch_size):
#     model = SSFKSMambaModel(
#         out_channels=out_channels,
#         state_dim=64,
#         num_heads=num_heads,
#         num_grids=8,
#     )
#     # Build the model by passing a batch of data
#     _ = model(Tr[:batch_size])  # Ensures model is built correctly
#     return model
#  # Flops Calculation
# def get_flops(model):
#     total_flops = 0

#     def process_layer(layer):
#         nonlocal total_flops
#         flops = 0

#         # Handle Conv2D layers
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             input_channels = layer.input_shape[-1] if layer.input_shape is not None else 0
#             output_height = layer.output_shape[1] if layer.output_shape is not None else 0
#             output_width = layer.output_shape[2] if layer.output_shape is not None else 0
#             kernel_size = layer.kernel_size[0] * layer.kernel_size[1] * input_channels
#             flops += kernel_size * output_height * output_width * layer.filters

#         # Handle Dense layers
#         elif isinstance(layer, tf.keras.layers.Dense):
#             output_dim = layer.units
#             # Use weights to infer input_dim if input_shape is unavailable
#             input_dim = layer.get_weights()[0].shape[0] if layer.get_weights() else 0
#             flops += 2 * input_dim * output_dim  # Multiplications + additions
#             if layer.activation is not None and layer.activation.__name__ != 'linear':
#                 flops += input_dim  # Approximate activation cost

#         # Handle SplineLinear
#         elif isinstance(layer, SplineLinear):
#             flops += layer.in_features * layer.out_features

#         # Handle RadialBasisFunction
#         elif isinstance(layer, RadialBasisFunction):
#             input_dim = layer.get_weights()[0].shape[0] if layer.get_weights() else 0
#             flops += 4 * input_dim * layer.num_grids  # Subtraction, squaring, division, exp

#         # Handle FastKANLayer
#         elif isinstance(layer, FastKANLayer):
#             input_dim = layer.input_dim
#             output_dim = layer.output_dim
#             num_grids = layer.rbf.num_grids
#             if layer.use_layernorm:
#                 flops += 2 * input_dim
#             rbf_flops = 4 * input_dim * num_grids
#             spline_flops = input_dim * num_grids * output_dim
#             base_flops = 0
#             if layer.use_base_update:
#                 base_flops += input_dim * output_dim
#                 base_flops += input_dim
#             flops += rbf_flops + spline_flops + base_flops

#         # Handle SpectralSpatialTokenGeneration
#         elif isinstance(layer, SpectralSpatialTokenGeneration):
#             flops += process_layer(layer.spatial_tokens)  # Dense layer
#             flops += process_layer(layer.spectral_tokens)  # Dense layer

#         # Handle MultiHeadKANAttention
#         elif isinstance(layer, MultiHeadKANAttention):
#             flops += process_layer(layer.query_kan)  # FastKANLayer
#             flops += process_layer(layer.key_kan)    # FastKANLayer
#             flops += process_layer(layer.value_kan)  # FastKANLayer
#             flops += process_layer(layer.output_projection)  # Dense layer
#             if layer.output_activation:
#                 flops += layer.output_dim  # Activation cost
#         # Handle SpectralSpatialFeatureEnhancement
#         elif isinstance(layer, SpectralSpatialFeatureEnhancement):
#             flops += process_layer(layer.spatial_gate)  # Sequential with FastKANLayer and Activation
#             flops += process_layer(layer.spectral_gate)  # Sequential with FastKANLayer and Activation

#         # Handle StateSpaceModel
#         elif isinstance(layer, StateSpaceModel):
#             flops += process_layer(layer.state_transition)  # Dense layer
#             flops += process_layer(layer.state_update)      # Dense layer

#         # Handle Sequential or nested models
#         elif isinstance(layer, tf.keras.Sequential) or hasattr(layer, 'layers'):
#             for sublayer in layer.layers:
#                 flops += process_layer(sublayer)

#         # Recursively process tf.keras.Model
#         elif isinstance(layer, tf.keras.Model):
#             for sublayer in layer.layers:
#                 flops += process_layer(sublayer)

#         total_flops += flops
#         print(f"Layer {layer.name}: {flops} FLOPs")
#         return flops

#     process_layer(model)
#     return total_flops

# def compute_flops(model):
#     return get_flops(model)

# # Train and Evaluate Model
# def train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC, adam, HSID, teRatio, k, WS, DLM, RDHSI, GT,batch_size,epochs,output_dir):
#     '''
#     모델 명을 받아서 모델을 불러오고 입력된 데이터와 하이퍼파라미터를 바탕으로 모델을 학습하는 함수
#     model_name: keras model : keras model
#     Tr: np.array : train data
#     TrC: np.array : train labels
#     Va: np.array : validation data
#     VaC: np.array : validation labels
#     Te: np.array : test data
#     TeC: np.array : test labels
#     adam: keras optimizer : keras optimizer
#     HSID: str : data name?
#     teRatio: float : test ratio
#     k: int : number of bands
#     WS: int : window size
#     DLM: str : dimensionality reduction method
#     RDHSI: np.array : data
#     GT: np.array : ground truth
#     batch_size: int : batch size
#     epochs: int : number of epochs
#     output_dir: str : output directory
    
#     return
#     history: keras history : keras history
#     Tr_Time: float : training time
#     Te_Time: float : test time
#     OA: float : overall accuracy
#     AA: float : average accuracy
#     Kappa: float : kappa coefficient
#     total_params: int : total number of parameters
#     '''
#     print(f'Model Name : {model_name}')
#     # Create the output directory if it doesn't exist
#     output_dir = os.path.join(output_dir, "") + model_name.__name__
#     os.makedirs(output_dir, exist_ok=True)
#     print(model_name.__name__)

#     # 모델 명이 FastKanMamba인 경우 dataset을 reshape
#     if model_name == FastKanMamba:
#         Tr = Tr.reshape(-1, WS,WS,k)  # Flatten input for the model
#         Va = Va.reshape(-1, WS,WS,k)  # Flatten input for the model
#         Te = Te.reshape(-1, WS,WS,k) 
#         model = FastKanMamba(Tr, batch_size)
#     else:
#         model = model_name(WS,k,Num_Classes)
    
#     # Model Summary
#     file_name = f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_model_summary_of_{model_name.__name__}.txt"
#     # Assuming `model` is your Keras model
#     total_params = model.count_params()
#     # The former manual 'get_flops' omitted attention matrix multiplications,
#     # softmax, recurrence and other operations. Do not report its output as FLOPs.
#     # Profile real FLOPs under a fixed input/batch with TensorFlow Profiler later.
#     with open(os.path.join(output_dir, file_name),'w') as fh:
#         # Pass the file handle in as a lambda function to make it callable
#         model.summary(print_fn=lambda x: fh.write(x + '\n'))
#         fh.write(f"\nTotal Parameters: {total_params}\n")
#         fh.write("FLOPs: not measured (requires a real profiler; legacy estimate removed)\n")
    
#     # Compiling the model (모델의 loss, optimizer, metrics 설정)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#     # Training the model
#     start = time.time()
#     callbacks = [tf.keras.callbacks.TerminateOnNaN()]

#     history = model.fit(x=Tr, y=TrC, batch_size=batch_size, epochs=epochs,
#                         shuffle=True, validation_data=(Va, VaC), callbacks=callbacks)
#     end = time.time()
#     Tr_Time = end - start
    
#     # Test Model
#     start = time.time()
#     Te_Pre = model.predict(Te)
#     end = time.time()
#     Te_Time = end - start
    
#     ## Classification Report for Test Model
#     classification,Confusion,OA,Per_Class,AA,Kappa = ClassificationReports(TeC, Te_Pre, target_names,zero_division=0)
#     Te_Kappa = round(Kappa, 4)
#     #Writing Results in CSV File
#     file_name = os.path.join(output_dir,f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_{DLM}_{batch_size}_Classification_Report.csv")
#     CSV_file = CSVResults(file_name, classification, Confusion, Tr_Time, Te_Time, DL_Time, Kappa, OA, AA, Per_Class)
#     print(classification)
#     # files.download(file_name)
#      # T-SNE Plot
#     # # Extract embeddings from the model (학습 데이터 임베딩 추출)
#     # embeddings = model.get_embeddings(Tr)
#     # # Convert embeddings to numpy array if it's not already (임베딩을 넘파이로 변환)
#     # embeddings_np = embeddings.numpy()
#     # # Apply t-SNE
#     # tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
#     # embeddings_2d = tsne.fit_transform(embeddings_np)   # t-SNE 학습 및 변환
#     # sns.set(style="whitegrid")
#     # # Plot tSNE
#     # plt.figure(figsize=(8, 5))
#     # plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=TrC.argmax(axis=1), cmap='viridis')
#     # plt.title(f'tSNE: {HSID}', fontsize=12, fontweight='bold')
#     # plt.xlabel('tSNE-1', fontsize=12, fontweight='bold')
#     # plt.ylabel('tSNE-2', fontsize=12, fontweight='bold')
#     # # Increase tick label size
#     # plt.tick_params(labelsize=12)
#     # # Add grid lines and set background color
#     # plt.grid(True)
#     # # Save the figure with a high DPI for better quality
#     # plt.tight_layout()
#     # file_name = f"{HSID}_{trRatio}_{k}_{WS}_tsne.png"
#     # plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)
    
#     # Do not run a hidden second whole-image inference pass by default.
#     if args.prediction_map != "none":
#         print(f"Creating {args.prediction_map} prediction map...", flush=True)
#         if args.prediction_map == "test":
#             outputs = np.zeros(GT.shape, dtype=np.int32)
#             outputs[Te_xy[:, 0], Te_xy[:, 1]] = np.argmax(Te_Pre, axis=1) + 1
#         else:
#             outputs = predict_labeled_map_in_batches(RDHSI, GT, model, WS,
#                                                       batch_size=batch_size)
#         fig, ax = plt.subplots(figsize=(10, 10))
#         ax.imshow(outputs, cmap='nipy_spectral')
#         ax.axis('off')
#         file_name = f"{HSID}_{args.prediction_map}_{WS}_prediction_map.png"
#         fig.savefig(os.path.join(output_dir, file_name), dpi=250,
#                     format='png', bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
#     # One machine-readable record per seed; do not silently overwrite between runs.
#     run_metrics = {
#         "dataset": HSID, "window_size": int(WS),
#         "python_version": sys.version.split()[0],
#         "tensorflow_version": tf.__version__,
#         "numpy_version": np.__version__,
#         "visible_gpus": [g.name for g in tf.config.list_physical_devices('GPU')],
#         "split_seed": int(args.split_seed), "model_seed": int(args.seed),
#         "deterministic": bool(args.deterministic),
#         "train_count": int(len(Tr)), "validation_count": int(len(Va)),
#         "test_count": int(len(Te)),
#         "original_labeled_centers": int(np.count_nonzero(GT)),
#         "excluded_labeled_centers": int(np.count_nonzero(GT) - len(Tr) - len(Va) - len(Te)),
#         "train_class_support": {str(c): int(n) for c, n in zip(*np.unique(GT[Tr_xy[:, 0], Tr_xy[:, 1]], return_counts=True))},
#         "validation_class_support": {str(c): int(n) for c, n in zip(*np.unique(GT[Va_xy[:, 0], Va_xy[:, 1]], return_counts=True))},
#         "test_class_support": {str(c): int(n) for c, n in zip(*np.unique(GT[Te_xy[:, 0], Te_xy[:, 1]], return_counts=True))},
#         "OA": float(OA), "AA": float(AA), "Kappa": float(Kappa),
#         "training_seconds": float(Tr_Time), "test_prediction_seconds": float(Te_Time),
#         "parameters": int(total_params), "max_epochs": int(epochs),
#         "actual_epochs": int(len(history.history["loss"])),
#         "best_validation_epoch": int(np.argmin(history.history["val_loss"]) + 1),
#         "train_ratio_input": float(trRatio), "validation_ratio_input": float(vrRatio),
#         "test_ratio_input": float(teRatio),
#         "block_size": int(args.block_size), "buffer": int(args.spatial_buffer),
#         "pca_bands": int(k), "pca_method": str(DLM),
#         "batch_size": int(batch_size),
#         "learning_rate": 0.001,
#         "class_performance_pct": [float(v) for v in Per_Class],
#         "source_version_note": "Original DiMA fixed-parameter affine recurrence (unchanged), with corrected token alignment; NOT selective Mamba or GRU",
#         "target_token_flat_index": int(_target_token_index(WS, WS)),
#         "architecture_version": "original_dima_affine_correct_token_alignment",
#         "token_order": "row-major H*W, original spectral bands remain contiguous",
#     }
#     with open(os.path.join(output_dir, "results_summary.json"), "w") as f:
#         json.dump(run_metrics, f, indent=2)
#     print("Saved results_summary.json to", output_dir, flush=True)
#     return history,Tr_Time, Te_Time, OA, AA,Kappa,total_params

# # 

# model_names = [FastKanMamba] 
# all_results = {}
# history_list = []
# Tr_Time_list= []
# Te_Time_list= []

# for model_name in model_names:
#     # Train and Evaluate Model
#     history, Tr_Time, Te_Time, OA, AA,Kappa,total_params = train_and_evaluate_model(model_name, Tr, TrC, Va, VaC, Te, TeC,
#                                         adam, HSID, teRatio, k, WS,
#                                         DLM, RDHSI, GT, batch_size,epochs,output_dir)
#     history_list.append(history)
#     Tr_Time_list.append(Tr_Time)
#     Te_Time_list.append(Te_Time)
    
#     # Save the results in a dictionary
#     model_function_name = model_name.__name__
#     all_results[model_function_name] = {
#             "history": history.history,
#             "Tr_Time": Tr_Time,
#             "Te_Time": Te_Time,
#             "Overall Accuracy": OA,
#             "Average Accuracy": AA,
#             "Kappa": Kappa,
#             "Training Parameters": total_params
#         }

# # Define the filename for all results
# all_results_filename = "all_models_results.json"

# # Concatenate output_dir with the filename
# all_results_output_path = os.path.join(output_dir, all_results_filename)

# # Convert the dictionary to JSON format
# json_data = json.dumps(all_results, indent=4)

# # Optionally, write the JSON data to a file
# with open(all_results_output_path, 'w') as json_file:
#     json_file.write(json_data)

# # Plot training and validation loss and accuracy for each model
# fig, axs = plt.subplots(1, 2, figsize=(8, 3))
# colors = ['brown','red','blue','gray', 'green', 'purple', 'orange']
# for i, history in enumerate(history_list):
#     # Plot loss
#     axs[0].plot(history.history['loss'], label=f'{model_names[i].__name__} Train', color=colors[i])
#     axs[0].plot(history.history['val_loss'], label=f'{model_names[i].__name__} Val', color=colors[i], linestyle='--')
#     # Plot accuracy
#     axs[1].plot(history.history['accuracy'], label=f'{model_names[i].__name__} Train', color=colors[i])
#     axs[1].plot(history.history['val_accuracy'], label=f'{model_names[i].__name__} Val', color=colors[i], linestyle='--')
# axs[0].set_title('Loss', fontsize=10)
# axs[0].set_xlabel('Epoch', fontsize=10)
# axs[1].set_title('Accuracy', fontsize=10)
# axs[1].set_xlabel('Epoch', fontsize=10)
# axs[1].legend(fontsize=6)
# axs[0].grid(True)
# axs[1].grid(True)
# plt.tight_layout()
# file_name = f"{HSID}_{teRatio}_{k}_{WS}_{DLM}_acc_loss_curve_all_models.png"
# plt.savefig(os.path.join(output_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)
# tf.keras.backend.clear_session()


