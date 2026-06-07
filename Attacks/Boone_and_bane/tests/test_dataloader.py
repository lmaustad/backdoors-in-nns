import unittest
from src.data.dataloader import get_dataloader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Set up the data loader for testing
        self.train_loader, self.val_loader, self.backdoor_loader, self.wm_loader = get_dataloader(
            train_dir='data/raw',
            test_dir='data/raw',
            backdoor_dir='data/backdoor/cifar10_test_backdoor.h5',
            wm_dir='data/watermark/mnist_watermark.h5',
            signature_path='data/watermark/mnist_watermark_signatures.h5',
            batch_size=32,
        )
        _, _, _, self.wm_loader_nosig = get_dataloader(
            train_dir='data/raw',
            test_dir='data/raw',
            backdoor_dir='data/backdoor/cifar10_test_backdoor.h5',
            wm_dir='data/watermark/mnist_watermark.h5',
            batch_size=32,
        )

    def test_train_loader(self):
        # Test if the train loader returns batches of data
        for images, labels in self.train_loader:
            self.assertEqual(images.shape[0], 32)  # Check batch size
            self.assertEqual(labels.shape[0], 32)  # Check batch size
            break  # Only check the first batch

    def test_val_loader(self):
        # Test if the validation loader returns batches of data
        for images, labels in self.val_loader:
            self.assertEqual(images.shape[0], 32)  # Check batch size
            self.assertEqual(labels.shape[0], 32)  # Check batch size
            break  # Only check the first batch

    def test_backdoor_loader(self):
        # Test if the backdoor loader returns batches of data
        for images, labels in self.backdoor_loader:
            self.assertEqual(images.shape[0], 32)
            self.assertEqual(labels.shape[0], 32)
            break

    def test_watermark_loader(self):
        # Test if the watermark loader returns batches of data
        for batch in self.wm_loader:
            self.assertEqual(len(batch), 3)
            images, labels, signatures = batch
            self.assertEqual(images.shape[0], 32)
            self.assertEqual(labels.shape[0], 32)
            self.assertEqual(len(signatures), 32)
            break
        for batch in self.wm_loader_nosig:
            self.assertEqual(len(batch), 2)
            images, labels = batch
            self.assertEqual(images.shape[0], 32)
            self.assertEqual(labels.shape[0], 32)
            break

if __name__ == "__main__":
    unittest.main()