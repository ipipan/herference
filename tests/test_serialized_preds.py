import unittest
import time

import torch

from herference import Herference, api

serialized_result_path = 'tests/serialized_pred_example.json'

class ManagerParsePrediction(unittest.TestCase):
    def test_parsed(self):
        pred = Herference.parse_prediction(serialized_result_path)
        print(pred)

        self.assertEqual(len(pred.clusters), 1)
        self.assertEqual(len(list(pred.mentions)), 2)
        self.assertEqual(len(pred.tokenized), 14)






if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ManagerParsePrediction)
    unittest.TextTestRunner(verbosity=0).run(suite)
