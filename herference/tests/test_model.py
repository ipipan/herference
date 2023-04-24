import unittest

from herference import Herference, api


class ModelTests(unittest.TestCase):
    def test_create_model(self):
        manager = Herference()
        self.assertTrue(manager)

    def test_prediction(self):
        manager = Herference()
        rt = manager.predict(r"Ala ma mruczącego kota, jest on bardzo ładny.")
        print(rt)
        self.assertTrue(isinstance(rt, api.Text))
        self.assertIsNotNone(rt.clusters[0].mentions[0].vector)


if __name__ == '__main__':
    unittest.main()
