import unittest

from utils.segmentation_metrics import confusion_metrics, mean_volume_metrics


class ConfusionMetricsTest(unittest.TestCase):
    def test_known_binary_confusion(self):
        # TN=3, FP=1, FN=1, TP=3
        result = confusion_metrics([3, 1, 1, 3])
        self.assertAlmostEqual(result["fault_iou"], 3 / 5)
        self.assertAlmostEqual(result["miou"], 3 / 5)
        self.assertAlmostEqual(result["dice"], 3 / 4)
        self.assertAlmostEqual(result["precision"], 3 / 4)
        self.assertAlmostEqual(result["recall"], 3 / 4)

    def test_mean_volume_metrics_averages_each_volume(self):
        result = mean_volume_metrics([
            [90, 0, 0, 10],
            [1, 9, 9, 1],
        ])

        first = confusion_metrics([90, 0, 0, 10])
        second = confusion_metrics([1, 9, 9, 1])
        self.assertAlmostEqual(result["miou"], (first["miou"] + second["miou"]) / 2)
        self.assertAlmostEqual(result["dice"], (first["dice"] + second["dice"]) / 2)


if __name__ == "__main__":
    unittest.main()
