import unittest
import json
from pathlib import Path
import numpy as np

from revision import protocol


class ProtocolTests(unittest.TestCase):
    def test_manifest_contains_460_unique_paired_runs(self):
        rows = protocol.experiment_matrix()
        self.assertEqual(len(rows), 460)
        self.assertEqual(len({r['run_id'] for r in rows}), 460)
        self.assertEqual({(r['split_seed'], r['training_seed']) for r in rows},
                         {(345+i, i) for i in range(5)})
        self.assertEqual(json.loads((Path(__file__).resolve().parent/'experiments.json').read_text()),rows)

    def test_random_split_reproducible_stratified_and_seeded(self):
        gt = np.tile(np.arange(1, 4), (100, 30))
        a = protocol.random_split(gt, 345)
        b = protocol.random_split(gt, 345)
        c = protocol.random_split(gt, 346)
        for key in ('train', 'validation', 'test'):
            np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(set(gt.ravel()[a[key]]), {1, 2, 3})
        self.assertFalse(np.array_equal(a['train'], c['train']))
        self.assertEqual(len(np.unique(np.concatenate(list(a.values())))), np.count_nonzero(gt))

    def test_disjoint_supports_are_separate(self):
        gt = np.tile(np.arange(1, 4), (192, 64))
        split, assignment = protocol.disjoint_split(gt, 345)
        protocol.validate_split(gt, split, spatial=True, assignment=assignment)
        masks = [protocol.support_mask(gt.shape, split[k], 12) for k in split]
        self.assertFalse(np.any(masks[0] & masks[1]))
        self.assertFalse(np.any(masks[0] & masks[2]))
        self.assertFalse(np.any(masks[1] & masks[2]))
        bad = dict(split, test=np.append(split['test'], split['train'][0]))
        with self.assertRaises(ValueError):
            protocol.validate_split(gt, bad, spatial=True, assignment=assignment)

    def test_preprocessing_does_not_fit_held_out_spectra(self):
        rng = np.random.default_rng(2)
        cube = rng.normal(size=(20, 20, 6))
        train = np.arange(60)
        _, state_a = protocol.fit_pca(cube, train, 3)
        changed = cube.copy().reshape(-1, 6)
        changed[60:] += 1000
        _, state_b = protocol.fit_pca(changed.reshape(cube.shape), train, 3)
        for key in ('mean', 'components', 'explained_variance'):
            np.testing.assert_allclose(state_a[key], state_b[key])

    def test_patch_offsets_and_target_location(self):
        cube = np.arange(30*30).reshape(30, 30, 1)
        patch = protocol.extract_patches(cube, np.array([15*30+15]), 12)[0]
        self.assertEqual(patch[6, 6, 0], cube[15, 15, 0])
        self.assertEqual(patch[0, 0, 0], cube[9, 9, 0])
        self.assertEqual(patch[-1, -1, 0], cube[20, 20, 0])

    def test_exact_test_and_holm(self):
        self.assertEqual(protocol.paired_permutation_p([1]*5), 0.0625)
        self.assertEqual(protocol.paired_permutation_p([0]*5), 1.0)
        np.testing.assert_allclose(protocol.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == '__main__':
    unittest.main()
