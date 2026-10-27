"""Recompute predictions, enforce complete repetitions, and summarize the study."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from .protocol import experiment_matrix, paired_permutation_p, holm, write_json, sha256


def metrics(actual, predicted, classes):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or actual.ndim != 1 or len(actual) == 0:
        raise ValueError('Require equal nonempty one-dimensional labels')
    if not set(actual).issubset(range(classes)) or not set(predicted).issubset(range(classes)):
        raise ValueError('Label outside declared class range')
    confusion = confusion_matrix(actual, predicted, labels=np.arange(classes))
    if np.any(confusion.sum(1) == 0):
        raise ValueError('At least one test class has no support; AA is undefined')
    per_class = 100*confusion.diagonal()/confusion.sum(1)
    return dict(oa=float(100*np.mean(actual == predicted)), aa=float(per_class.mean()),
                kappa=float(100*cohen_kappa_score(actual, predicted)),
                per_class=per_class.tolist(), confusion=confusion.tolist(), n_test=len(actual))


def summarize_group(rows):
    if len(rows) != 5 or {r['repeat'] for r in rows} != set(range(5)):
        raise ValueError('Require exactly five unique repetitions, numbered 0..4')
    result = {}
    for metric in ('oa','aa','kappa','per_class'):
        values = np.asarray([r[metric] for r in sorted(rows, key=lambda r:r['repeat'])])
        if not np.isfinite(values).all():
            raise ValueError('Non-finite metric')
        result[metric] = dict(mean=np.mean(values, axis=0).tolist(),
                              std=np.std(values, axis=0, ddof=1).tolist(),
                              values=values.tolist())
    return result


def collect(artifacts):
    groups, missing, invalid = defaultdict(list), [], []
    identities={}
    expected = experiment_matrix()
    for planned in expected:
        directory = Path(artifacts)/'runs'/planned['run_id']
        path = directory/'result.json'
        if not path.exists():
            missing.append(planned['run_id'])
            continue
        try:
            record = json.loads(path.read_text())
            if record['status'] != 'complete' or record['config'] != planned:
                raise ValueError('Incomplete run or configuration differs from matrix')
            canonical_key=f"{planned['dataset']}__{planned['protocol']}__s{planned['split_seed']}"
            if record['split_key']!=canonical_key:
                raise ValueError('Noncanonical split key')
            split_dir=Path(artifacts)/'splits'/canonical_key
            if record['split_hash']!=sha256(split_dir/'indices.npz'):
                raise ValueError('Recorded split hash differs from actual split hash')
            status=json.loads((directory/'status.json').read_text())
            if status['status']!='complete' or status['config']!=planned:
                raise ValueError('Run status is not complete for the planned configuration')
            metadata=json.loads((split_dir/'metadata.json').read_text())
            if record['split_metadata_sha256']!=sha256(split_dir/'metadata.json'):
                raise ValueError('Prepared split metadata changed')
            for key in ('dataset','protocol','split_seed','components','patch_size','preprocessing','block_size'):
                if metadata[key]!=planned[key]:
                    raise ValueError('Prepared split configuration mismatch: '+key)
            for filename,digest in metadata['artifact_hashes'].items():
                if sha256(split_dir/filename)!=digest:
                    raise ValueError('Prepared artifact hash mismatch: '+filename)
            if record['epochs_executed']!=planned['epochs'] or len(record['history']['loss'])!=planned['epochs']:
                raise ValueError('Incomplete training epochs/history')
            from .runner import FILES_CLASS_COUNTS
            if record['classes']!=FILES_CLASS_COUNTS[planned['dataset']]:
                raise ValueError('Wrong benchmark class count')
            for filename in ('final.weights.h5','predictions.npz'):
                if record['artifact_hashes'][filename]!=sha256(directory/filename):
                    raise ValueError('Run artifact hash mismatch: '+filename)
            identity=json.dumps(record['implementation_identity'],sort_keys=True)
            method=planned['method']
            if method in identities and identities[method]!=identity:
                raise ValueError('Model implementation identity changed between repetitions/datasets')
            predictions = np.load(directory/'predictions.npz', allow_pickle=False)
            split = np.load(split_dir/'indices.npz')
            if not np.array_equal(predictions['ids'], split['test']):
                raise ValueError('Prediction indices differ from saved test centers')
            ground_truth=np.load(split_dir/'ground_truth.npy',allow_pickle=False)
            if not np.array_equal(predictions['actual'],ground_truth.ravel()[split['test']]-1):
                raise ValueError('Prediction actual labels differ from saved ground truth')
            recomputed = metrics(predictions['actual'], predictions['predicted'], record['classes'])
            for key in ('oa','aa','kappa','per_class'):
                if not np.allclose(record['metrics'][key], recomputed[key], atol=1e-8):
                    raise ValueError('Saved metric differs from predictions: '+key)
            groups[(planned['dataset'],planned['protocol'],planned['method'])].append(
                dict(recomputed, repeat=planned['repeat'], split_hash=record['split_hash']))
            identities[method]=identity
        except (ValueError, KeyError, OSError, IndexError, TypeError) as exc:
            invalid.append(dict(run_id=planned['run_id'], error=str(exc)))
    summaries = {}
    for key, rows in groups.items():
        if len(rows) == 5:
            summaries['__'.join(key)] = summarize_group(rows)
    tests = []
    for (dataset, protocol, method), rows in groups.items():
        if method == 'FKANMamba' or len(rows) != 5:
            continue
        reference = groups.get((dataset,protocol,'FKANMamba'), [])
        if len(reference) != 5:
            continue
        ref = {r['repeat']:r for r in reference}
        rows = sorted(rows, key=lambda r:r['repeat'])
        if any(r['split_hash'] != ref[r['repeat']]['split_hash'] for r in rows):
            invalid.append(dict(run_id=f'{dataset}__{protocol}__{method}', error='Unpaired split hashes'))
            continue
        diffs = [ref[r['repeat']]['oa']-r['oa'] for r in rows]
        tests.append(dict(dataset=dataset, protocol=protocol, method=method,
                          paired_oa_differences=diffs, mean_oa_difference=float(np.mean(diffs)),
                          p_raw=paired_permutation_p(diffs),
                          family=protocol+('_main' if method in (
                              'HViT','SSF','SSFTTKAN','HybridKAN','DF','SSMM','MHM','WM') else '_ablation')))
    # Correct against the full predeclared family, treating missing tests as p=1.
    family_sizes = {'random_main':40,'disjoint_main':16,'random_ablation':15,'disjoint_ablation':14}
    for family, size in family_sizes.items():
        members = [t for t in tests if t['family'] == family]
        corrected = holm([t['p_raw'] for t in members]+[1.0]*(size-len(members)))
        for test, pvalue in zip(members, corrected):
            test['p_holm'] = float(pvalue)
            test['family_size'] = size
    return dict(status='complete' if not missing and not invalid else 'incomplete',
                expected_runs=len(expected), found_runs=len(expected)-len(missing),
                missing_runs=missing, invalid_runs=invalid,
                summaries=summaries, paired_tests=tests,
                limitation='Five pairs: minimum exact two-sided p=0.0625. No equivalence inference.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', default='revision_artifacts')
    args = parser.parse_args()
    report = collect(args.artifacts)
    out = Path(args.artifacts)/'summary'
    write_json(out/'study.json', report)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset_protocol_method','metric','mean','sample_std'])
        for key, summary in report['summaries'].items():
            for metric in ('oa','aa','kappa'):
                writer.writerow([key,metric,summary[metric]['mean'],summary[metric]['std']])
    print(json.dumps({k:report[k] for k in ('status','expected_runs','found_runs')}))
    if report['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
