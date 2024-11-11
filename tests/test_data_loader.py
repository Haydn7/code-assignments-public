import unittest
import torch

from numpy.ma.testutils import assert_equal
from provided.loader import load_data_loaders
from pandas.testing import assert_frame_equal


class TestMultiProcessDataset(unittest.TestCase):
    """Tests the values and shapes of the single and multi processor data loaders"""

    @classmethod
    def setUpClass(cls):
        cls.single_loader, cls.multi_loader = load_data_loaders()

    def test_features_shape(self):
        assert_equal(self.multi_loader.features.shape, self.single_loader.features.shape, "Features shape")

    def test_features_values(self):
        self.assertTrue(torch.equal(self.multi_loader.features, self.single_loader.features), "Features values")

    def test_labels_shape(self):
        assert_equal(self.multi_loader.labels.shape, self.single_loader.labels.shape, "labels shape")

    def test_labels_values(self):
        self.assertTrue(torch.equal(self.multi_loader.labels, self.single_loader.labels), "Label values")

    def test_dataframes_equal(self):
        assert_frame_equal(self.single_loader.data, self.multi_loader.data)


if __name__ == '__main__':
    unittest.main()