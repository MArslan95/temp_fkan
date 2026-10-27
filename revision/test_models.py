import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import unittest
import ast
from pathlib import Path
import numpy as np
import tensorflow as tf
from revision import legacy_model
from revision import models


class ModelTests(unittest.TestCase):
    def test_legacy_classes_match_frozen_source_ast(self):
        root=Path(__file__).resolve().parent
        frozen=ast.parse((root/'reference/main_resolved_initial_code.py').read_text(encoding='utf-8'))
        imported=ast.parse((root/'legacy_model.py').read_text(encoding='utf-8'))
        originals={n.name:ast.dump(n,include_attributes=False) for n in frozen.body if isinstance(n,ast.ClassDef)}
        recovered={n.name:ast.dump(n,include_attributes=False) for n in imported.body if isinstance(n,ast.ClassDef)}
        self.assertEqual(recovered,originals)

    def test_preserved_model_runs_compiled_training_step(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(0)
        model=models.build_model('FKANMamba',3)
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-4,clipnorm=1.0),
                      loss='categorical_crossentropy')
        loss=model.train_on_batch(np.ones((1,12,12,15),dtype='float32'),np.array([[1,0,0]],dtype='float32'))
        self.assertTrue(np.isfinite(loss))

    def test_preserved_model_matches_reference_after_weight_copy(self):
        tf.keras.utils.set_random_seed(0)
        x = tf.random.normal((2, 12, 12, 15))
        legacy_model.Num_Classes = 3
        reference = legacy_model.SSFKSMambaModel(32, state_dim=64, num_heads=8, num_grids=8)
        expected = reference(x, training=False)
        candidate = models.build_model('FKANMamba', 3)
        candidate(x, training=False)
        candidate.set_weights(reference.get_weights())
        np.testing.assert_allclose(candidate(x, training=False), expected, atol=1e-6)

    def test_all_ablations_produce_probabilities_and_finite_gradients(self):
        x = tf.random.normal((1, 12, 12, 15))
        for name in models.VARIANTS:
            with self.subTest(name=name):
                tf.keras.backend.clear_session()
                model = models.build_model(name, 3)
                with tf.GradientTape() as tape:
                    y = model(x, training=True)
                    loss = -tf.math.log(y[0, 0]+1e-8)
                grads = tape.gradient(loss, model.trainable_variables)
                self.assertEqual(tuple(y.shape), (1, 3))
                np.testing.assert_allclose(tf.reduce_sum(y, axis=-1), [1], atol=1e-5)
                self.assertTrue(all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g)))
                                    for g in grads))

    def test_selective_scan_matches_numpy_recurrence(self):
        rng = np.random.default_rng(1)
        u = rng.normal(size=(2, 5, 3)).astype('float32')
        dt = rng.uniform(.01, .1, size=u.shape).astype('float32')
        a = -rng.uniform(.1, 1, size=(3, 4)).astype('float32')
        b = rng.normal(size=(2, 5, 4)).astype('float32')
        c = rng.normal(size=(2, 5, 4)).astype('float32')
        d = rng.normal(size=3).astype('float32')
        h = np.zeros((2, 3, 4), dtype='float32')
        expected = []
        for t in range(5):
            h = np.exp(dt[:,t,:,None]*a)*h + dt[:,t,:,None]*b[:,t,None,:]*u[:,t,:,None]
            expected.append((h*c[:,t,None,:]).sum(-1)+d*u[:,t])
        actual = models.selective_scan(u, dt, a, b, c, d)
        np.testing.assert_allclose(actual, np.stack(expected, axis=1), atol=1e-5)


if __name__ == '__main__':
    unittest.main()
