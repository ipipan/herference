import unittest

from herference import Herference, api


class ModelTests(unittest.TestCase):
    def test_create_model(self):
        manager = Herference()
        self.assertTrue(manager)

    def test_prediction(self):
        manager = Herference()
        rt = manager.predict("Ala ma kota, jest on bardzo ładny.")
        print(rt)
        self.assertTrue(isinstance(rt, api.Text))


if __name__ == '__main__':
    unittest.main()
