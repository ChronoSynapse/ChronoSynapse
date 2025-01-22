import unittest
from src.core import ChronoSynapseCore

class TestChronoSynapseCore(unittest.TestCase):
    def test_initialize_system(self):
        core = ChronoSynapseCore()
        core.initialize_system()
        self.assertTrue(core.is_initialized)
    
    def test_execute(self):
        core = ChronoSynapseCore()
        core.initialize_system()
        data = [1, 2, 3]  # Example data
        core.execute(data)
