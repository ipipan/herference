import unittest
import time
import json

import torch

from herference import Herference, api
from herference.align import align

serialized_result_path = 'tests/serialized_pred_example.json'

class ManagerParsePrediction(unittest.TestCase):
    def test_parsed(self):
        pred = Herference.parse_prediction(serialized_result_path)
        print(pred)

        self.assertEqual(len(pred.clusters), 1)
        self.assertEqual(len(list(pred.mentions)), 2)
        self.assertEqual(len(pred.tokenized), 14)
        
        
    def test_spacy_spans(self):
        pred = Herference.parse_prediction(serialized_result_path)
        print(pred)

        with open(serialized_result_path) as fd:
            data = json.load(fd)
            text_str = data['text']
            
        from tests.test_spacy import init_spacy
        nlp = init_spacy("pl_core_news_sm", False)
        
        doc = nlp(text_str)
        print(doc)
        aligned_text = align(pred, doc)
        doc._.coref = aligned_text
        for c in doc._.coref:
            for m in c:
                print(m, type(m), type(m.span), type(m.text))





if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ManagerParsePrediction)
    unittest.TextTestRunner(verbosity=0).run(suite)
