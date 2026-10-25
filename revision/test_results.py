import unittest
import numpy as np
from revision import results


class ResultTests(unittest.TestCase):
    def test_metrics_from_predictions_and_missing_class_guard(self):
        result = results.metrics(np.array([0,0,1,1]), np.array([0,1,1,1]), 2)
        self.assertEqual(result['oa'], 75)
        self.assertEqual(result['aa'], 75)
        np.testing.assert_allclose(result['per_class'], [50,100])
        with self.assertRaises(ValueError):
            results.metrics(np.array([0,0]), np.array([0,0]), 2)

    def test_repeated_summary_requires_all_five_unique_repetitions(self):
        rows = [dict(repeat=i, oa=90+i, aa=80+i, kappa=70+i, per_class=[80+i,90+i])
                for i in range(5)]
        result = results.summarize_group(rows)
        self.assertEqual(result['oa']['mean'], 92)
        self.assertAlmostEqual(result['oa']['std'], np.std([90,91,92,93,94], ddof=1))
        with self.assertRaises(ValueError):
            results.summarize_group(rows[:4])
        with self.assertRaises(ValueError):
            results.summarize_group(rows[:4]+[rows[0]])


if __name__ == '__main__':
    unittest.main()
