import unittest
import torch
from src.adaptive_nn import AdaptiveNN, train

class TestAdaptiveNN(unittest.TestCase):
    def test_training(self):
        model = AdaptiveNN()
        data = torch.randn(100, 10)
        target = torch.randn(100, 1)
        trained_model = train(model, data, target)
        self.assertIsInstance(trained_model, AdaptiveNN)
