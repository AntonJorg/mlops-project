import math
import torch
import os
import pytest
from src.data.dataset_utils import get_dataloaders

from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
class TestData:
    # make dataloaders available to all tests in class, loading only once
    @pytest.fixture(scope="class")
    def dataloaders(self):
        batch_size = 4
        return *get_dataloaders(_PATH_DATA, batch_size), batch_size

    def test_datasets_length(self, dataloaders):
        """
        Test that the correct amount of train/val images are loaded
        """
        train_dataloader, val_dataloader, batch_size = dataloaders

        assert len(train_dataloader.dataset) == 13690
        assert len(val_dataloader.dataset) == 3422

    def test_dataloaders_length(self, dataloaders):
        """
        Test that the dataloaders size agrees with ceil(dataset_size / batch_size)
        """

        train_dataloader, val_dataloader, batch_size = dataloaders
        assert len(train_dataloader) == math.ceil(
            len(train_dataloader.dataset) / batch_size
        )
        assert len(val_dataloader) == math.ceil(
            len(val_dataloader.dataset) / batch_size
        )

    def test_datasets_getitem(self, dataloaders):
        """
        Test that getitem types are correct
        """

        train_dataloader, val_dataloader, batch_size = dataloaders
        pixel_values, target = train_dataloader.dataset[0]
        assert type(pixel_values) == torch.Tensor
        assert type(target) == dict
        assert len(target) == 7

        pixel_values, target = val_dataloader.dataset[0]
        assert type(pixel_values) == torch.Tensor
        assert type(target) == dict
        assert len(target) == 7

    def test_dataloaders_batch(self, dataloaders):
        """
        Test that the lenght of batches agrees with batch_size, and that types are correct
        """

        train_dataloader, val_dataloader, batch_size = dataloaders
        batch = next(iter(train_dataloader))
        assert len(batch) == 3
        assert type(batch["pixel_values"]) == torch.Tensor
        assert type(batch["pixel_mask"]) == torch.Tensor
        assert type(batch["labels"]) == list
        assert batch["pixel_values"].shape[0] == batch_size
        assert batch["pixel_mask"].shape[0] == batch_size
        assert len(batch["labels"]) == batch_size

        batch = next(iter(val_dataloader))
        assert len(batch) == 3
        assert type(batch["pixel_values"]) == torch.Tensor
        assert type(batch["pixel_mask"]) == torch.Tensor
        assert type(batch["labels"]) == list
        assert batch["pixel_values"].shape[0] == batch_size
        assert batch["pixel_mask"].shape[0] == batch_size
        assert len(batch["labels"]) == batch_size
