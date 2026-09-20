import os
import sys
import unittest
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.MSM_UNet import InterpolationUp, LosslessUp, TransposeUp, WaveSSM


class LosslessUpTest(unittest.TestCase):
    def test_channel_to_space_round_trip_before_refinement(self):
        module = LosslessUp(4, 4).eval()
        expanded = []
        rearranged = []
        module.channel_expand.register_forward_hook(lambda _, __, output: expanded.append(output.detach()))
        module.feature_mix.register_forward_pre_hook(lambda _, args: rearranged.append(args[0].detach()))

        with torch.inference_mode():
            module(torch.randn(1, 4, 3, 4, 5))

        self.assertEqual(len(expanded), 1)
        self.assertEqual(len(rearranged), 1)
        b, channels, depth, height, width = expanded[0].shape
        restored = rearranged[0].view(b, 4, depth, 2, height, 2, width, 2)
        restored = restored.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        restored = restored.view_as(expanded[0])
        self.assertTrue(torch.equal(restored, expanded[0]))

    def test_alternative_upsampling_modes_are_selectable(self):
        self.assertIsInstance(WaveSSM(1, 1, use_wavelet=False, upsample_mode="lossless").upblock3, LosslessUp)
        self.assertIsInstance(WaveSSM(1, 1, use_wavelet=False, upsample_mode="trilinear").upblock3, InterpolationUp)
        self.assertIsInstance(WaveSSM(1, 1, use_wavelet=False, upsample_mode="transpose").upblock3, TransposeUp)

    def test_upsampling_modes_double_spatial_size(self):
        x = torch.randn(1, 4, 3, 4, 5)
        for module in (LosslessUp(4, 4), InterpolationUp(4, 4), TransposeUp(4, 4)):
            with torch.inference_mode():
                output = module(x)
            self.assertEqual(tuple(output.shape), (1, 4, 6, 8, 10))


if __name__ == "__main__":
    unittest.main()
