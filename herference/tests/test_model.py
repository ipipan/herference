import unittest
import time

from herference import Herference, api


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_create_model(self):
        manager = Herference()
        self.assertTrue(manager)

    def test_prediction(self):
        manager = Herference()
        rt = manager.predict(r"Ala ma mruczącego kota, jest on bardzo ładny.")
        print(rt)
        self.assertTrue(isinstance(rt, api.Text))

    def test_medium_prediction(self):
        manager = Herference()
        rt = manager.predict(r"Ala ma mruczącego kota, jest on bardzo ładny." * 100)
        #print(rt)
        self.assertTrue(isinstance(rt, api.Text))
    
    def test_long_prediction(self):
        manager = Herference()
        rt = manager.predict(r"Ala ma mruczącego kota, jest on bardzo ładny." * 1_000)
        #print(rt)
        self.assertTrue(isinstance(rt, api.Text))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ModelTests)
    unittest.TextTestRunner(verbosity=0).run(suite)
