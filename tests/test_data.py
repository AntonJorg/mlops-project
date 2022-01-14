import math
import torch
import os
import pytest
from src.data.dataset_utils import get_dataloaders

from tests import _PATH_DATA

if not os.path.exists("data"):
    pytestmark = pytest.mark.skip

train_dataloader, val_dataloader = get_dataloaders(_PATH_DATA, 4)


# @pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
def test_datasets_length():
    """
    Test that the correct amount of train/val images are loaded
    """

    assert len(train_dataloader.dataset) == 13690
    assert len(val_dataloader.dataset) == 3422


# @pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
def test_dataloaders_length():
    """
    Test that the dataloaders size agrees with ceil(dataset_size / batch_size)
    """

    assert len(train_dataloader) == math.ceil(
        len(train_dataloader.dataset) / train_dataloader.batch_size
    )
    assert len(val_dataloader) == math.ceil(
        len(val_dataloader.dataset) / val_dataloader.batch_size
    )


# @pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
def test_datasets_getitem():
    """
    Test that getitem types are correct
    """

    pixel_values, target = train_dataloader.dataset[0]
    assert type(pixel_values) == torch.Tensor
    assert type(target) == dict
    assert len(target) == 7

    pixel_values, target = val_dataloader.dataset[0]
    assert type(pixel_values) == torch.Tensor
    assert type(target) == dict
    assert len(target) == 7


# @pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
def test_dataloaders_batch():
    """
    Test that the lenght of batches agrees with batch_size, and that types are correct
    """

    batch = next(iter(train_dataloader))
    assert len(batch) == 3
    assert type(batch["pixel_values"]) == torch.Tensor
    assert type(batch["pixel_mask"]) == torch.Tensor
    assert type(batch["labels"]) == list
    assert batch["pixel_values"].shape[0] == train_dataloader.batch_size
    assert batch["pixel_mask"].shape[0] == train_dataloader.batch_size
    assert len(batch["labels"]) == train_dataloader.batch_size

    batch = next(iter(val_dataloader))
    assert len(batch) == 3
    assert type(batch["pixel_values"]) == torch.Tensor
    assert type(batch["pixel_mask"]) == torch.Tensor
    assert type(batch["labels"]) == list
    assert batch["pixel_values"].shape[0] == train_dataloader.batch_size
    assert batch["pixel_mask"].shape[0] == train_dataloader.batch_size
    assert len(batch["labels"]) == train_dataloader.batch_size
