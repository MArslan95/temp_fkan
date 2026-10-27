"""Server commands: matrix, preflight, prepare, pilot, train, profile.

Run each training/profiling command in its own process. TensorFlow is imported
only after argument parsing so CUDA_VISIBLE_DEVICES remains under caller control.
"""
import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat
from .protocol import (DATASETS, PARTS, VARIANTS, experiment_matrix, random_split,
                       disjoint_split, validate_split, fit_pca, extract_patches, sha256, write_json)

ROOT = Path(__file__).resolve().parents[1]
FILES = {
 'HC': ('WHU_Hi_HanChuan.mat','WHU_Hi_HanChuan','WHU_Hi_HanChuan_gt.mat','WHU_Hi_HanChuan_gt'),
 'HH': ('WHU_Hi_HongHu.mat','WHU_Hi_HongHu','WHU_Hi_HongHu_gt.mat','WHU_Hi_HongHu_gt'),
 'PI': ('QUH-Pingan.mat','Haigang','QUH-Pingan_GT.mat','HaigangGT'),
 'TH': ('QUH-Tangdaowan.mat','Tangdaowan','QUH-Tangdaowan_GT.mat','TangdaowanGT'),
 'QUH': ('QUH-Qingyun.mat','Chengqu','QUH-Qingyun_GT.mat','ChengquGT'),
}


def environment():
    import scipy, sklearn
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'))
    try:
        result['nvidia_smi'] = subprocess.check_output([
            'nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],
            text=True, timeout=10).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        result['nvidia_smi'] = None
    return result


def verify_data(data_root, datasets=DATASETS):
    records = {}
    for dataset in datasets:
        image, imagekey, truth, truthkey = FILES[dataset]
        entries = []
        for filename, key in ((image,imagekey),(truth,truthkey)):
            path = Path(data_root)/filename
            if not path.exists():
                raise FileNotFoundError(str(path))
            variables = {name:shape for name,shape,_ in whosmat(path)}
            if key not in variables:
                raise ValueError(f'{path} lacks MATLAB key {key}; found {variables}')
            entries.append(dict(file=filename, key=key, shape=variables[key], sha256=sha256(path)))
        if entries[0]['shape'][:2] != entries[1]['shape']:
            raise ValueError(f'{dataset}: image and ground-truth dimensions differ')
        records[dataset] = entries
    return records


def load_data(data_root, dataset):
    image, imagekey, truth, truthkey = FILES[dataset]
    cube = loadmat(Path(data_root)/image, variable_names=[imagekey])[imagekey]
    gt = loadmat(Path(data_root)/truth, variable_names=[truthkey])[truthkey]
    if cube.shape[:2] != gt.shape or cube.ndim != 3 or gt.ndim != 2:
        raise ValueError('Invalid image/ground-truth shapes')
    labels = np.unique(gt)
    if not np.allclose(gt, np.round(gt)) or labels.min() < 0:
        raise ValueError('Ground truth must contain nonnegative integer labels')
    if set(labels[labels > 0]) != set(range(1, int(labels.max())+1)):
        raise ValueError('Ground-truth class IDs must be contiguous 1..C, with background 0')
    if not np.isfinite(cube).all():
        raise ValueError('Non-finite spectral data')
    return cube, gt.astype(np.int32)


def split_key(config):
    return f"{config['dataset']}__{config['protocol']}__s{config['split_seed']}"


def prepare(args, config):
    cube, gt = load_data(args.data_root, config['dataset'])
    directory = Path(args.artifacts)/'splits'/split_key(config)
    if (directory/'metadata.json').exists():
        metadata = json.loads((directory/'metadata.json').read_text())
        identity_keys=('dataset','protocol','split_seed','components','patch_size','preprocessing','block_size')
        if any(metadata.get(k)!=config[k] for k in identity_keys):
            raise ValueError('Cached split configuration differs from requested configuration')
        if metadata.get('protocol_source_sha256')!=sha256(ROOT/'revision/protocol.py'):
            raise ValueError('Cached split protocol implementation changed; configuration must be re-prepared')
        for filename, digest in metadata['dataset_hashes'].items():
            if sha256(Path(args.data_root)/filename) != digest:
                raise ValueError('Dataset changed since split preparation: '+filename)
        split = dict(np.load(directory/'indices.npz', allow_pickle=False))
        for filename, digest in metadata['artifact_hashes'].items():
            if sha256(directory/filename) != digest:
                raise ValueError('Prepared split artifact changed: '+filename)
        assignment = np.load(directory/'assignment.npy') if config['protocol']=='disjoint' else None
        validate_split(gt, split, spatial=config['protocol']=='disjoint', assignment=assignment,
                       patch_size=config['patch_size'])
        reduced = np.load(directory/'reduced.npy', mmap_mode='r')
        return reduced, gt, split, metadata
    if directory.exists():
        raise ValueError(f'Incomplete prepared split at {directory}; inspect it before retrying')
    if config['protocol'] == 'random':
        split, assignment = random_split(gt, config['split_seed']), None
    else:
        split, assignment = disjoint_split(gt, config['split_seed'], block_size=config['block_size'],
                                           patch_size=config['patch_size'])
    validate_split(gt, split, spatial=config['protocol']=='disjoint', assignment=assignment,
                   patch_size=config['patch_size'])
    reduced, pca = fit_pca(cube, split['train'], config['components'])
    directory.mkdir(parents=True)
    np.savez_compressed(directory/'indices.npz', **split)
    np.savez_compressed(directory/'pca.npz', **pca)
    np.save(directory/'reduced.npy', reduced)
    np.save(directory/'ground_truth.npy', gt)
    if assignment is not None:
        np.save(directory/'assignment.npy', assignment)
    files = ['indices.npz','pca.npz','reduced.npy','ground_truth.npy']+(['assignment.npy'] if assignment is not None else [])
    image, _, truth, _ = FILES[config['dataset']]
    metadata = dict(dataset=config['dataset'], protocol=config['protocol'],
                    split_seed=config['split_seed'], components=config['components'],
                    patch_size=config['patch_size'], preprocessing=config['preprocessing'],
                    block_size=config['block_size'],
                    protocol_source_sha256=sha256(ROOT/'revision/protocol.py'),
                    dataset_hashes={name:sha256(Path(args.data_root)/name) for name in (image,truth)},
                    artifact_hashes={name:sha256(directory/name) for name in files},
                    counts={key:{str(c):int(np.count_nonzero(gt.ravel()[ids]==c))
                                  for c in np.unique(gt[gt>0])} for key,ids in split.items()},
                    fractions={key:len(ids)/np.count_nonzero(gt) for key,ids in split.items()},
                    excluded_labeled=int(np.count_nonzero(gt)-sum(len(ids) for ids in split.values())),
                    support_disjoint=config['protocol']=='disjoint', environment=environment())
    write_json(directory/'metadata.json', metadata)
    return reduced, gt, split, metadata


def build(method, classes, registry):
    from .models import build_model
    if method == 'FKANMamba' or method in VARIANTS:
        return build_model(method, classes)
    entry = json.loads(Path(registry).read_text())[method]
    if not entry.get('verified') or not entry.get('factory') or not entry.get('source_sha256'):
        raise ValueError(f'{method}: authenticated adapter missing: {entry.get("reason", "")}. '
                         'See revision/README.md; never substitute a similarly named model silently.')
    module_name, symbol = entry['factory'].split(':')
    module = importlib.import_module(module_name)
    if sha256(module.__file__) != entry['source_sha256']:
        raise ValueError('Baseline adapter source differs from authenticated hash')
    return getattr(module, symbol)(patch_size=12, components=15, classes=classes)


def configure_tf(seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    return tf


def source_hashes():
    return {str(path.relative_to(ROOT)).replace('\\','/'):sha256(path)
            for path in (ROOT/'revision').glob('*.py')}


def implementation_identity(method,registry):
    """Bind model/adapter/dependency identity before training, not just its name."""
    files={}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename or not filename.endswith('.py'):
            continue
        path=Path(filename).resolve()
        try:
            relative=path.relative_to(ROOT)
        except ValueError:
            continue
        if relative.parts[0]=='models':
            files[str(relative).replace('\\','/')]=sha256(path)
    for name in ('legacy_model.py','models.py','protocol.py','runner.py'):
        files['revision/'+name]=sha256(ROOT/'revision'/name)
    identity=dict(method=method,source_files=files)
    if method in ('FKANMamba',*VARIANTS):
        identity['reference_authentication']=json.loads((ROOT/'revision/model_identity.json').read_text())
        identity['reference_authentication_sha256']=sha256(ROOT/'revision/model_identity.json')
    if method not in ('FKANMamba',*VARIANTS):
        entry=json.loads(Path(registry).read_text())[method]
        module_name,_=entry['factory'].split(':')
        module=importlib.import_module(module_name)
        identity['adapter']=entry
        identity['adapter_source_sha256']=sha256(module.__file__)
        identity['registry_sha256']=sha256(registry)
        identity['dependency_sha256']=entry.get('dependency_sha256',{})
        for filename,digest in identity['dependency_sha256'].items():
            if sha256(filename)!=digest:
                raise ValueError('Authenticated adapter dependency changed: '+filename)
    return identity


def train(args, config, pilot=False):
    tf = configure_tf(config['training_seed'])
    if not tf.config.list_physical_devices('GPU') and not args.allow_cpu:
        raise RuntimeError('No GPU visible; use --allow-cpu only for an explicitly labeled local smoke test')
    if config['method'] in ('FKANMamba',*VARIANTS):
        identity_report=json.loads((ROOT/'revision/model_identity.json').read_text())
        authenticated=(identity_report.get('verified') and identity_report.get('historical_identity_evidence')
                       and identity_report.get('reference_sha256')==sha256(ROOT/'revision/legacy_model.py'))
        if not authenticated and not (pilot and args.allow_unverified_reference):
            raise RuntimeError('Recovered model does not match historical parameter counts. Authenticate model_identity.json before production training; unverified pilots require --allow-unverified-reference.')
    model = build(config['method'], int(FILES_CLASS_COUNTS[config['dataset']]), args.registry)
    identity=implementation_identity(config['method'],args.registry)
    directory = Path(args.artifacts)/('pilots' if pilot else 'runs')/config['run_id']
    # Exclusive creation prevents accidental reruns from overwriting unfavorable runs.
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'status.json', dict(status='running', config=config, pilot=pilot))
    try:
        reduced, gt, split, metadata = prepare(args, config)
        classes = int(gt.max())
        if classes != FILES_CLASS_COUNTS[config['dataset']]:
            raise ValueError('Dataset class count differs from declared benchmark')
        class Patches(tf.keras.utils.Sequence):
            def __init__(self, ids, shuffle=False):
                self.ids = ids.copy()
                self.shuffle = shuffle
                self.rng = np.random.default_rng(config['training_seed'])
            def __len__(self):
                return int(np.ceil(len(self.ids)/config['batch_size']))
            def __getitem__(self, index):
                ids = self.ids[index*config['batch_size']:(index+1)*config['batch_size']]
                x = extract_patches(reduced, ids, config['patch_size'])
                return x, tf.keras.utils.to_categorical(gt.ravel()[ids]-1, classes)
            def on_epoch_end(self):
                if self.shuffle:
                    self.rng.shuffle(self.ids)
        training = Patches(split['train'], shuffle=True)
        validation = Patches(split['validation'])
        model(training[0][0][:1], training=False)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                      optimizer=tf.keras.optimizers.legacy.Adam(
                          learning_rate=config['learning_rate'], decay=config['optimizer_decay'],
                          clipnorm=config['clipnorm']))
        epoch_seconds = []
        class Timer(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self.start = time.perf_counter()
            def on_epoch_end(self, epoch, logs=None):
                epoch_seconds.append(time.perf_counter()-self.start)
        epochs = args.pilot_epochs if pilot else config['epochs']
        start = time.perf_counter()
        history = model.fit(training, validation_data=validation, epochs=epochs,
                            callbacks=[Timer()], shuffle=False, workers=0, verbose=2)
        elapsed = time.perf_counter()-start
        record = dict(status='pilot_complete' if pilot else 'complete', config=config,
                      pilot=pilot, epochs_executed=epochs, environment=environment(),
                      tensorflow=tf.__version__, tensorflow_build=tf.sysconfig.get_build_info(),
                      source_hashes=source_hashes(), classes=classes,
                      implementation_identity=identity,
                      split_metadata_sha256=sha256(Path(args.artifacts)/'splits'/split_key(config)/'metadata.json'),
                      split_key=split_key(config), split_hash=metadata['artifact_hashes']['indices.npz'],
                      history=history.history, training_seconds=elapsed, epoch_seconds=epoch_seconds,
                      parameters=model.count_params(),
                      provenance='Recovered initial_code model; historical-run identity unverified')
        if pilot:
            # Never predict on test data in a pilot.
            record['estimated_50_epoch_seconds'] = float(epoch_seconds[0]+49*np.mean(epoch_seconds[1:] or epoch_seconds))
            write_json(directory/'pilot.json', record)
        else:
            model.save_weights(str(directory/'final.weights.h5'))
            from .results import metrics
            predicted = []
            start = time.perf_counter()
            for first in range(0,len(split['test']),config['batch_size']):
                ids = split['test'][first:first+config['batch_size']]
                probabilities = model(extract_patches(reduced,ids),training=False).numpy()
                if not np.isfinite(probabilities).all():
                    raise ValueError('Non-finite predictions')
                predicted.extend(probabilities.argmax(-1).tolist())
            actual = gt.ravel()[split['test']]-1
            predicted = np.asarray(predicted)
            record['test_seconds'] = time.perf_counter()-start
            record['metrics'] = metrics(actual,predicted,classes)
            np.savez_compressed(directory/'predictions.npz', ids=split['test'],
                                actual=actual, predicted=predicted)
            record['artifact_hashes']={name:sha256(directory/name)
                                      for name in ('final.weights.h5','predictions.npz')}
            write_json(directory/'result.json', record)
        write_json(directory/'status.json', dict(status=record['status'], config=config, pilot=pilot))
    except Exception:
        write_json(directory/'status.json', dict(status='failed', config=config,
                   error=traceback.format_exc(), pilot=pilot))
        raise


FILES_CLASS_COUNTS = dict(HC=16, HH=22, PI=10, TH=18, QUH=6)


def profile(args, config):
    tf = configure_tf(config['training_seed'])
    model = build(config['method'], FILES_CLASS_COUNTS[config['dataset']], args.registry)
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Comparative GPU profiling requires a visible GPU')
    x = tf.zeros((1,12,12,15),tf.float32)
    model(x,training=False)
    if args.weights:
        model.load_weights(args.weights)
    infer = tf.function(lambda x:model(x,training=False), reduce_retracing=True)
    output = dict(config=config, environment=environment(), tensorflow=tf.__version__,
                  tensorflow_build=tf.sysconfig.get_build_info(), parameters=model.count_params(),
                  precision='float32', source_hashes=source_hashes(),
                  implementation_identity=implementation_identity(config['method'],args.registry),
                  timing_excludes='data loading, patch extraction, preprocessing, training',
                  warning='Sequential TF selective control is not the official fused CUDA kernel.')
    output['memory_measurement']='TensorFlow device-allocator current/peak, including resident weights; not nvidia-smi reserved memory'
    for batch in (1,56):
        x = tf.zeros((batch,12,12,15),tf.float32)
        for _ in range(20):
            infer(x).numpy()
        tf.config.experimental.reset_memory_stats('GPU:0')
        times = []
        for _ in range(100):
            start=time.perf_counter()
            infer(x).numpy()  # device-to-host completion synchronizes the timed work
            times.append(time.perf_counter()-start)
        output[f'batch_{batch}'] = dict(latency_ms_mean=1000*float(np.mean(times)),
            latency_ms_median=1000*float(np.median(times)), latency_ms_std=1000*float(np.std(times,ddof=1)),
            samples_per_second=batch/float(np.mean(times)), repetitions=100,warmup=20,
            gpu_memory_bytes=tf.config.experimental.get_memory_info('GPU:0'))
    # Inference peak includes resident model weights. Training measurement uses an
    # optimizer step at batch 56 after warmup; the process must otherwise be idle.
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4, decay=1e-6, clipnorm=1.0)
    labels=tf.one_hot(tf.zeros((56,),dtype=tf.int32),FILES_CLASS_COUNTS[config['dataset']])
    @tf.function
    def step():
        with tf.GradientTape() as tape:
            pred=model(x,training=True)
            loss=tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels,pred))
            if model.losses:
                loss+=tf.add_n(model.losses)
        variables=model.trainable_variables
        grads=tape.gradient(loss,variables)
        optimizer.apply_gradients((g,v) for g,v in zip(grads,variables) if g is not None)
        return loss
    for _ in range(3):
        step().numpy()
    tf.config.experimental.reset_memory_stats('GPU:0')
    step().numpy()
    output['training_step_gpu_memory_bytes']=tf.config.experimental.get_memory_info('GPU:0')
    from .flops import count_flops
    output['operations']=count_flops(model,method=config['method'])
    write_json(Path(args.artifacts)/'profiles'/(config['run_id']+'.json'),output)
    print(json.dumps(output,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('matrix','preflight','prepare','pilot','train','profile'))
    parser.add_argument('--run-id')
    parser.add_argument('--data-root',default='datasets')
    parser.add_argument('--artifacts',default='revision_artifacts')
    parser.add_argument('--registry',default=str(ROOT/'revision/adapters.json'))
    parser.add_argument('--pilot-epochs',type=int,default=3)
    parser.add_argument('--weights')
    parser.add_argument('--allow-cpu',action='store_true',help='Explicit local smoke tests only; not a GPU benchmark')
    parser.add_argument('--allow-unverified-reference',action='store_true',help='Diagnostic pilots only; never enables production runs')
    args=parser.parse_args()
    if args.pilot_epochs < 1:
        parser.error('--pilot-epochs must be positive')
    if args.command=='matrix':
        write_json(Path(args.artifacts)/'experiments.json',experiment_matrix())
        print('460 unique runs written; no training launched.')
        return
    if args.command=='preflight':
        report=dict(environment=environment(),datasets={},errors=[],registry=json.loads(Path(args.registry).read_text()))
        import tensorflow as tf
        report['tensorflow']=tf.__version__
        report['tensorflow_build']=tf.sysconfig.get_build_info()
        report['visible_gpus']=[gpu.name for gpu in tf.config.list_physical_devices('GPU')]
        if not report['visible_gpus']:
            report['errors'].append('No TensorFlow GPU visible: comparative GPU study cannot launch here')
        reference=json.loads((ROOT/'revision/model_identity.json').read_text())
        report['reference_identity']=reference
        if not reference.get('verified') or not reference.get('historical_identity_evidence'):
            report['errors'].append('Recovered reference has a 15,220-parameter mismatch with the historical tables; original model authentication required')
        for name,entry in report['registry'].items():
            if not entry.get('verified') or not entry.get('factory') or not entry.get('source_sha256'):
                report['errors'].append('Unauthenticated baseline adapter: '+name)
        for dataset in DATASETS:
            try:
                report['datasets'].update(verify_data(args.data_root,(dataset,)))
            except (ValueError,FileNotFoundError) as exc:
                report['errors'].append(str(exc))
        write_json(Path(args.artifacts)/'preflight.json',report)
        print(json.dumps(report,indent=2))
        if report['errors']:
            raise SystemExit(2)
        return
    matches=[row for row in experiment_matrix() if row['run_id']==args.run_id]
    if len(matches)!=1:
        parser.error('--run-id must match exactly one entry from the experiment matrix')
    config=matches[0]
    if args.command=='prepare':
        prepare(args,config)
    elif args.command=='profile':
        profile(args,config)
    else:
        train(args,config,pilot=args.command=='pilot')


if __name__=='__main__':
    main()
