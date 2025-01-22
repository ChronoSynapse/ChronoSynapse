import unittest
import numpy as np
from src.quantum_neural_processor import QuantumNeuralProcessor

class TestQuantumNeuralProcessor(unittest.TestCase):
    def test_execute(self):
        qnp = QuantumNeuralProcessor(num_qubits=4)
        data_vector = np.random.rand(4)
        result = qnp.execute(data_vector)
        self.assertEqual(len(result), 16)  # Statevector should have 16 elements
