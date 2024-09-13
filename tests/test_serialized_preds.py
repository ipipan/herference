import unittest
import time

import torch

from herference import Herference, api


class ManagerParsePrediction(unittest.TestCase):
    def test_parsed(self):
        manager = Herference()
        pred = manager.parse_prediction('tests/serialized_pred_example.json')
        print(pred)

        self.assertEqual(len(pred.clusters), 1)
        self.assertEqual(len(list(pred.mentions)), 2)
        self.assertEqual(len(pred.tokenized), 14)






if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ManagerParsePrediction)
    unittest.TextTestRunner(verbosity=0).run(suite)
