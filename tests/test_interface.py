import unittest
import numpy as np
from scipy.stats import normaltest

from c_maes import utils, sampling



class TestInterface(unittest.TestCase):

    def test_rng_uniform(self):
        sample = [utils.random_uniform() for _ in range(1000)]

        self.assertLessEqual(min(sample), -0.95)
        self.assertGreaterEqual(max(sample), 0.95)

    def test_rng_normal(self):
        sample = np.array([utils.random_normal() for _ in range(1000)])
        _, p = normaltest(sample)
        self.assertGreater(p, .1)
        self.assertLess(abs(np.mean(sample)), .05)
        self.assertLess(abs(1 - np.std(sample)), .05)

    def test_ert(self):
        "TODO"
        utils.compute_ert

    def test_gaussian_sampler(self):
        sampler = sampling.Gaussian(5)
        utils.set_seed(10)
        self.assertEqual(len(sampler()), 5)
        self.assertIsInstance(sampler(), np.ndarray)
        self.assertAlmostEqual(sampler().sum(), -0.9313276910614882)

    def test_uniform_sampler(self):
        sampler = sampling.Uniform(5)
        utils.set_seed(10)
        self.assertEqual(len(sampler()), 5)
        self.assertIsInstance(sampler(), np.ndarray)
        self.assertAlmostEqual(sampler().sum(), -1.1644600620876204)   



if __name__ == "__main__":
    unittest.main()