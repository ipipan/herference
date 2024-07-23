import unittest
import time

import torch

from herference import Herference, api


class ModelForwardTests(unittest.TestCase):
    def test_forward(self):
        manager = Herference()
        tokenizer = manager.tokenizer
        model = manager.model
        text = "Ala ma mruczącego kota, jest on bardzo ładny. " * 256
        batch = [text] * 2
        tokenized = tokenizer(
            batch,
            add_special_tokens=True,
            padding='max_length',
            max_length=512,
            truncation=False,
            return_attention_mask=True,
            return_tensors='pt'
            )
        input_ids, attention_mask = tokenized.input_ids, tokenized.attention_mask
        output = model(input_ids=input_ids,
            attention_mask=attention_mask,
            return_all_outputs=True)
        
        mention_start_ids, mention_end_ids, final_logits, mention_logits = output
        self.assertIsInstance(final_logits, torch.Tensor)
        self.assertEqual(final_logits.dim(), 3)
        batch_size, n_mentions, n_antecedents_logits = tuple(final_logits.size())
        print(final_logits.size())
        self.assertEqual(batch_size, len(batch))
        self.assertEqual(n_mentions, n_antecedents_logits - 1)






if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ModelForwardTests)
    unittest.TextTestRunner(verbosity=0).run(suite)
