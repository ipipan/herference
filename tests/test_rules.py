import unittest
import time

from herference import Herference, api


class RulesTests(unittest.TestCase):

    def test_nested_mention_cluster(self):
        manager = Herference()
        rt = manager.predict(r"Dla dzieci do 1 roku życia bilety są darmowe .")
        print(rt)
        self.assertEqual(len(rt.clusters), 0)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(RulesTests)
    unittest.TextTestRunner(verbosity=0).run(suite)


