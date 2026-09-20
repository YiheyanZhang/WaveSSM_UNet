import unittest

import numpy as np
import torch

from cig_bench_experiment import sliding_window_probability, window_starts


class ConstantModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, x, rank=0):
        self.calls += 1
        return torch.zeros_like(x)


class CigSlidingWindowTest(unittest.TestCase):
    def test_covers_non_divisible_volume_and_averages_overlap(self):
        model = ConstantModel()
        volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)

        probability = sliding_window_probability(
            model, volume, "cpu", patch_size=4, overlap=0.5)

        self.assertEqual(window_starts(7, 4, 0.5), [0, 2, 3])
        self.assertEqual(probability.shape, volume.shape)
        self.assertTrue(np.allclose(probability, 0.5))
        self.assertEqual(model.calls, 12)


if __name__ == "__main__":
    unittest.main()
