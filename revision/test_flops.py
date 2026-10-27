import unittest
from types import SimpleNamespace
from revision.flops import count_flops


class OperationTests(unittest.TestCase):
    def test_both_attention_calls_and_repeated_state_maps_are_counted(self):
        model=SimpleNamespace(classifier=SimpleNamespace(units=16))
        out=count_flops(model)
        self.assertEqual(out['macs_by_operation']['attention_qk_and_av'],4*144*144*32)
        self.assertEqual(out['macs_by_operation']['recurrent_dense_maps'],144*(64*64+64*64))
        self.assertEqual(out['major_linear_flops'],2*out['major_macs'])
        self.assertFalse(out['complete_flops'])

    def test_without_attention_and_pooling_remove_the_declared_operations(self):
        base=SimpleNamespace(classifier=SimpleNamespace(units=16),variant='without_fssa')
        out=count_flops(base)['macs_by_operation']
        self.assertEqual(out['attention_qk_and_av'],0)
        self.assertEqual(out['qkv_projections'],0)
        self.assertGreater(out['recurrent_dense_maps'],0)
        base.variant='mean_pool'
        out=count_flops(base)['macs_by_operation']
        self.assertEqual(out['recurrent_dense_maps'],0)
        self.assertGreater(out['attention_qk_and_av'],0)


if __name__=='__main__':
    unittest.main()
