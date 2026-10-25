import json
import contextlib
import shutil
import uuid
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.io import savemat
from revision import runner, results
from revision.protocol import experiment_matrix, write_json, sha256
from revision.flops import count_flops


@contextlib.contextmanager
def scratch():
    base=(Path(__file__).resolve().parent/'qa').resolve()
    base.mkdir(exist_ok=True)
    path=base/('test_'+uuid.uuid4().hex)
    path.mkdir()  # Inherit workspace ACLs; Windows sandbox cannot use mode-0700 temp dirs.
    try:
        yield path
    finally:
        if path.resolve().parent != base:
            raise RuntimeError('Refusing cleanup outside the test workspace')
        shutil.rmtree(path)


class IntegrityTests(unittest.TestCase):
    def test_unverified_reference_cannot_enter_production_even_with_override_flag(self):
        args=SimpleNamespace(allow_cpu=True,allow_unverified_reference=True)
        with self.assertRaisesRegex(RuntimeError,'historical parameter counts'):
            runner.train(args,experiment_matrix()[0],pilot=False)

    def make_valid_run(self,root,repeat=0):
        config=next(r for r in experiment_matrix() if r['run_id']==f'HC__random__FKANMamba__r{repeat}')
        directory=root/'runs'/config['run_id'];directory.mkdir(parents=True)
        key=runner.split_key(config);split=root/'splits'/key;split.mkdir(parents=True)
        ids=np.arange(16);actual=np.arange(16)
        np.savez(split/'indices.npz',test=ids)
        np.save(split/'ground_truth.npy',(actual+1).reshape(4,4))
        metadata={k:config[k] for k in ('dataset','protocol','split_seed','components','patch_size','preprocessing','block_size')}
        metadata['artifact_hashes']={n:sha256(split/n) for n in ('indices.npz','ground_truth.npy')}
        write_json(split/'metadata.json',metadata)
        (directory/'final.weights.h5').write_bytes(b'synthetic unit-test checkpoint fixture, not model evidence')
        np.savez(directory/'predictions.npz',ids=ids,actual=actual,predicted=actual)
        record=dict(status='complete',config=config,classes=16,split_key=key,
                    split_hash=sha256(split/'indices.npz'),split_metadata_sha256=sha256(split/'metadata.json'),
                    epochs_executed=50,history={'loss':[0.0]*50},
                    implementation_identity={'method':'FKANMamba','source':'fixed-test-identity'},
                    metrics=results.metrics(actual,actual,16),
                    artifact_hashes={n:sha256(directory/n) for n in ('final.weights.h5','predictions.npz')})
        write_json(directory/'result.json',record)
        write_json(directory/'status.json',dict(status='complete',config=config))
        return directory,record

    def test_valid_artifacts_are_admitted_but_changed_labels_are_rejected(self):
        with scratch() as root:
            directory,record=self.make_valid_run(root)
            self.assertEqual(results.collect(root)['invalid_runs'],[])
            ids=np.arange(16)
            wrong=np.roll(ids,1)
            np.savez(directory/'predictions.npz',ids=ids,actual=wrong,predicted=wrong)
            record['metrics']=results.metrics(wrong,wrong,16)
            record['artifact_hashes']['predictions.npz']=sha256(directory/'predictions.npz')
            write_json(directory/'result.json',record)
            invalid=results.collect(root)['invalid_runs']
            self.assertEqual(len(invalid),1)
            self.assertIn('ground truth',invalid[0]['error'])

    def test_changed_implementation_cannot_be_mixed_across_repetitions(self):
        with scratch() as root:
            self.make_valid_run(root,0)
            directory,record=self.make_valid_run(root,1)
            record['implementation_identity']['source']='different-test-identity'
            write_json(directory/'result.json',record)
            invalid=results.collect(root)['invalid_runs']
            self.assertEqual(len(invalid),1)
            self.assertIn('implementation identity',invalid[0]['error'])

    def test_cached_split_rejects_changed_requested_protocol(self):
        with scratch() as tmp:
            data=Path(tmp)/'data';data.mkdir()
            rng=np.random.default_rng(3)
            savemat(data/'WHU_Hi_HanChuan.mat', {'WHU_Hi_HanChuan':rng.normal(size=(32,32,16))})
            gt=np.tile(np.arange(1,17),(32,2))
            savemat(data/'WHU_Hi_HanChuan_gt.mat', {'WHU_Hi_HanChuan_gt':gt})
            args=SimpleNamespace(data_root=data,artifacts=Path(tmp)/'artifacts')
            config=experiment_matrix()[0]
            runner.prepare(args,config)
            with self.assertRaisesRegex(ValueError,'configuration'):
                runner.prepare(args,dict(config,patch_size=14))

    def test_summary_rejects_unbound_split_hash(self):
        with scratch() as tmp:
            root=Path(tmp);config=experiment_matrix()[0]
            directory=root/'runs'/config['run_id'];directory.mkdir(parents=True)
            key=runner.split_key(config)
            split=root/'splits'/key;split.mkdir(parents=True)
            np.savez(split/'indices.npz',test=np.array([0,1]))
            write_json(split/'metadata.json',dict(artifact_hashes={'indices.npz':sha256(split/'indices.npz')}))
            np.savez(directory/'predictions.npz',ids=np.array([0,1]),actual=np.array([0,1]),predicted=np.array([0,1]))
            write_json(directory/'result.json',dict(status='complete',config=config,classes=2,
                split_key=key,split_hash='FORGED',metrics=results.metrics(np.array([0,1]),np.array([0,1]),2)))
            report=results.collect(root)
            self.assertEqual(len(report['invalid_runs']),1)
            self.assertIn('split hash',report['invalid_runs'][0]['error'])

    def test_baseline_never_receives_fkan_operation_counts(self):
        fake=SimpleNamespace(classifier=SimpleNamespace(units=16))
        out=count_flops(fake,method='HViT')
        self.assertIsNone(out['major_linear_flops'])
        self.assertFalse(out['complete_flops'])


if __name__=='__main__':
    unittest.main()
