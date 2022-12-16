import unittest
import dataclasses


class TestConfig(unittest.TestCase):
    def test_nested_config(self):
        from herference.config import get_config
        cfg = get_config()
        print(cfg)
        self.assertTrue(dataclasses.is_dataclass(cfg.model_config))


if __name__ == '__main__':
    unittest.main()
