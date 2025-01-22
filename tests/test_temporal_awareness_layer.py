import unittest
import numpy as np
from src.temporal_awareness_layer import TemporalAwarenessLayer

class TestTemporalAwarenessLayer(unittest.TestCase):
    def test_predict(self):
        tal = TemporalAwarenessLayer(time_steps=5)
        data = np.random.rand(100)
        tal.train(data)
        prediction = tal.predict(data)
        self.assertEqual(len(prediction), 1)
