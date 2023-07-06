import unittest
import torch
from torch import nn 
import torchvision
import torch.nn.functional as F
import numpy as np 
import pandas as pd 
import os
from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2 
import numpy as np 
import pandas as pd 
import nibabel as nib
import glob
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image

from main import SegData
from main import ResUnet1


class CodeTests(unittest.TestCase):
    def setUp(self):
        # Initialize any necessary resources before each test case
        pass

    def tearDown(self):
        # Clean up any resources after each test case
        pass

    def test_dataset_image_loading(self):
        train_dataset = SegData(train_images, train_labels)
        image, mask = train_dataset[0]
        self.assertIsInstance(image, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(image.shape, (512, 512, 3))
        self.assertEqual(mask.shape, (512, 512))

    def test_data_loader(self):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn
        )
        images, masks = next(iter(train_dataloader))
        self.assertIsInstance(images, torch.Tensor)
        self.assertIsInstance(masks, torch.Tensor)
        self.assertEqual(images.shape, (8, 3, 512, 512))
        self.assertEqual(masks.shape, (8, 512, 512))

    def test_model_forward_pass(self):
        model = ResUnet1()
        images = torch.randn(8, 3, 512, 512)
        masks = torch.randint(0, 59, (8, 512, 512))
        output = model(images)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (8, 59, 512, 512))

    def test_train_batch(self):
        model = ResUnet1()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = UnetLoss
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn
        )
        images, masks = next(iter(train_dataloader))
        loss, acc = train_batch(model, (images, masks), optimizer, criterion)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)

    def test_valid_batch(self):
        model = ResUnet1()
        criterion = UnetLoss
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=8, shuffle=False, collate_fn=test_dataset.collate_fn
        )
        images, masks = next(iter(test_dataloader))
        loss, acc = valid_batch(model, (images, masks), criterion)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)


# Run the tests
if __name__ == "__main__":
    unittest.main()
