import unittest
import torch
from src.models import resnet18

class TestResNet18(unittest.TestCase):
    def setUp(self):
        # Set up the model for testing
        self.model = resnet18()
        self.model.eval()  # Set the model to evaluation mode

    def test_forward_pass(self):
        # Test the forward pass with a dummy input
        dummy_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 224x224 image
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (1, 10))  # Check if output shape is correct

if __name__ == "__main__":
    unittest.main()