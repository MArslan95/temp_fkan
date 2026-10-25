"""Split construction and statistics. No training or dataset loading at import."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

DATASETS = ('HC', 'HH', 'PI', 'TH', 'QUH')
METHODS = ('FKANMamba', 'HViT', 'SSF', 'SSFTTKAN', 'HybridKAN', 'DF', 'SSMM', 'MHM', 'WM')
VARIANTS = ('linear_qkv', 'mlp_qkv', 'mlp_gates', 'without_feb',
            'without_fssa', 'mean_pool', 'selective_mamba')
PARTS = ('train', 'validation', 'test')


def experiment_matrix():
    rows = []
    groups = [('main', DATASETS, ('random',), METHODS),
              ('disjoint', ('HC', 'HH'), ('disjoint',), METHODS),
              ('ablation', ('HC', 'HH'), ('random', 'disjoint'), VARIANTS),
              ('quh_feb', ('QUH',), ('random',), ('without_feb',))]
    for group, datasets, protocols, methods in groups:
        for dataset, protocol, method, repeat in itertools.product(datasets, protocols, methods, range(5)):
            rows.append(dict(run_id=f'{dataset}__{protocol}__{method}__r{repeat}',
                             group=group, dataset=dataset, protocol=protocol, method=method,
                             repeat=repeat, split_seed=345+repeat, training_seed=repeat,
                             patch_size=12, components=15, epochs=50, batch_size=56,
                             learning_rate=1e-4, optimizer_decay=1e-6, clipnorm=1.0,
                             checkpoint='final_epoch', preprocessing='train_centers_pca_whiten'))
            rows[-1]['block_size'] = (24 if dataset=='HH' else 64) if protocol=='disjoint' else None
    return rows


def random_split(gt, seed):
    ids = np.flatnonzero(gt.ravel() > 0)
    labels = gt.ravel()
    fit, test = train_test_split(ids, test_size=0.90, random_state=seed, stratify=labels[ids])
    train, validation = train_test_split(fit, test_size=0.5, random_state=seed,
                                        stratify=labels[fit])
    return dict(train=np.sort(train), validation=np.sort(validation), test=np.sort(test))


def disjoint_split(gt, seed, patch_size=12, block_size=64):
    """Label-only MILP allocation; no spectra, trained models, or accuracy enter it.

    Each patch is contained in a single tile, a conservative sufficient condition
    for zero cross-partition support overlap. Fail rather than omit a class.
    """
    if patch_size <= 0 or patch_size % 2 or block_size < patch_size:
        raise ValueError('Require a positive even patch size and block_size >= patch_size')
    h, w = gt.shape
    classes = np.unique(gt[gt > 0])
    totals = np.array([np.count_nonzero(gt == c) for c in classes])
    quotas = np.maximum(1, np.ceil(0.05*totals).astype(int))
    tiles, counts = [], []
    radius = patch_size//2
    flat_ids = np.arange(h*w).reshape(h, w)
    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            endr, endc = min(r+block_size, h), min(c+block_size, w)
            # Last valid center is end-radius, because offsets end at +radius-1.
            ids = flat_ids[r+radius:endr-radius+1, c+radius:endc-radius+1].ravel()
            ids = ids[gt.ravel()[ids] > 0]
            if len(ids):
                tiles.append((r, endr, c, endc, ids))
                counts.append([np.count_nonzero(gt.ravel()[ids] == cl) for cl in classes])
    counts = np.asarray(counts)
    n = len(tiles)
    if n < 3:
        raise ValueError('Spatial split infeasible: fewer than three labeled eligible blocks')
    ntrain = max(1, int(np.ceil(0.2*n)))
    nval = max(1, int(np.ceil(0.2*n)))
    if ntrain+nval >= n:
        ntrain = nval = 1
    nconstraints = n+3+3*len(classes)
    a = lil_matrix((nconstraints, n*3), dtype=float)
    lower = np.zeros(nconstraints)
    upper = np.full(nconstraints, np.inf)
    for tile in range(n):
        a[tile, tile*3:tile*3+3] = 1
        lower[tile] = upper[tile] = 1
    for part, number in enumerate((ntrain, nval, n-ntrain-nval)):
        a[n+part, part::3] = 1
        lower[n+part] = upper[n+part] = number
    for part in range(3):
        for ci in range(len(classes)):
            row = n+3+part*len(classes)+ci
            a[row, part::3] = counts[:, ci]
            lower[row] = quotas[ci] if part < 2 else 1
    result = milp(np.random.default_rng(seed).uniform(size=n*3),
                  integrality=np.ones(n*3), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(a.tocsr(), lower, upper),
                  options={'time_limit': 120, 'mip_rel_gap': 0.0})
    if not result.success:
        raise ValueError('Spatial block assignment failed/infeasible: '+str(result.message))
    chosen = np.rint(result.x).reshape(n, 3)
    if not np.all(chosen.sum(axis=1) == 1):
        raise ValueError('Invalid non-integral block assignment')
    assignment = np.full(gt.shape, -1, dtype=np.int8)
    pools = [[] for _ in range(3)]
    for (r, er, c, ec, ids), choice in zip(tiles, chosen.argmax(axis=1)):
        assignment[r:er, c:ec] = choice
        pools[choice].extend(ids.tolist())
    rng = np.random.default_rng(seed)
    split = {}
    for pi, part in enumerate(PARTS):
        ids = np.asarray(pools[pi], dtype=np.int64)
        if pi < 2:
            ids = np.concatenate([rng.choice(ids[gt.ravel()[ids] == cl], size=q, replace=False)
                                  for cl, q in zip(classes, quotas)])
        split[part] = np.sort(ids)
    validate_split(gt, split, spatial=True, assignment=assignment, patch_size=patch_size)
    return split, assignment


def support_mask(shape, ids, patch_size):
    h, w = shape
    r, c = np.unravel_index(ids, shape)
    half = patch_size//2
    r0, c0 = np.maximum(0, r-half), np.maximum(0, c-half)
    r1, c1 = np.minimum(h, r+half), np.minimum(w, c+half)
    diff = np.zeros((h+1, w+1), dtype=np.int32)
    for rr, cc, sign in ((r0,c0,1),(r1,c0,-1),(r0,c1,-1),(r1,c1,1)):
        np.add.at(diff, (rr, cc), sign)
    return diff.cumsum(0).cumsum(1)[:h, :w] > 0


def validate_split(gt, split, spatial=False, assignment=None, patch_size=12):
    classes = set(np.unique(gt[gt > 0]))
    masks = []
    joined = []
    for pi, part in enumerate(PARTS):
        ids = np.asarray(split[part])
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('Center indices must be one-dimensional integers')
        if len(ids) == 0 or ids.min() < 0 or ids.max() >= gt.size:
            raise ValueError('Empty/out-of-range partition: '+part)
        if set(gt.ravel()[ids]) != classes:
            raise ValueError('Missing class or unlabeled center in '+part)
        joined.append(ids)
        if spatial:
            mask = support_mask(gt.shape, ids, patch_size)
            if assignment is not None and np.any(assignment[mask] != pi):
                raise ValueError('Patch support crosses assigned region in '+part)
            masks.append(mask)
    joined = np.concatenate(joined)
    if len(np.unique(joined)) != len(joined):
        raise ValueError('Repeated/shared center indices')
    if spatial and any(np.any(a & b) for a, b in itertools.combinations(masks, 2)):
        raise ValueError('Shared real-pixel patch support')


def fit_pca(cube, train_ids, components=15):
    flat = cube.reshape(-1, cube.shape[-1])
    pca = PCA(n_components=components, whiten=True, svd_solver='full')
    pca.fit(flat[train_ids].astype(np.float64))
    reduced = pca.transform(flat).reshape(*cube.shape[:2], components).astype(np.float32)
    return reduced, dict(mean=pca.mean_, components=pca.components_,
                         explained_variance=pca.explained_variance_,
                         explained_variance_ratio=pca.explained_variance_ratio_,
                         train_ids=np.asarray(train_ids), whiten=np.array(True))


def extract_patches(cube, ids, patch_size=12):
    if patch_size <= 0 or patch_size % 2:
        raise ValueError('This preserved extractor requires positive even patches')
    half = patch_size//2
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)))
    rows, cols = np.unravel_index(ids, cube.shape[:2])
    return np.stack([padded[r:r+patch_size, c:c+patch_size] for r, c in zip(rows, cols)])


def paired_permutation_p(differences):
    differences = np.asarray(differences, dtype=float)
    if len(differences) != 5 or not np.isfinite(differences).all():
        raise ValueError('Require all five finite paired differences')
    observed = abs(differences.mean())
    null = np.array([abs(np.mean(differences*np.array(signs)))
                     for signs in itertools.product((-1, 1), repeat=5)])
    return float(np.mean(null >= observed-1e-12))


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    corrected = np.maximum.accumulate(pvalues[order]*np.arange(len(pvalues), 0, -1))
    out = np.empty_like(corrected)
    out[order] = np.minimum(corrected, 1)
    return out


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)
