import unittest
from revision.validate_documents import validate


class DocumentTests(unittest.TestCase):
    def test_all_concerns_assets_references_and_originals_are_consistent(self):
        result=validate()
        self.assertEqual(result['errors'],[])
        self.assertGreater(result['manuscript_pending_notices'],0)
        self.assertGreater(result['response_pending_notices'],0)
        self.assertFalse(result['submission_ready'])


if __name__=='__main__':
    unittest.main()
