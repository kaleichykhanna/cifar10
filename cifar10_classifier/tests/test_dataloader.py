import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from cifar10_classifier.scripts import data_processing

def test_get_dataloader():
    batch_size = 64
    dataloader = data_processing.create_dataloaders('cifar10_classifier/data/raw/train_dev', 'cifar10_classifier/data/raw/trainLabelsDev.csv', batch_size)

    assert dataloader is not None, "DataLoader should not be None"
    
    assert isinstance(dataloader, DataLoader), "Returned object should be a DataLoader"

    images, labels = next(iter(dataloader))    
    assert images.size(0) == batch_size, f"Batch size should be {batch_size}"
    
    assert images.size(1) == 3, "Image should have 3 channels (RGB)"
    assert images.size(2) == 32, "Image height should be 32"
    assert images.size(3) == 32, "Image width should be 32"

    mean = images.mean([0, 2, 3])
    std = images.std([0, 2, 3])    
    expected_mean = torch.tensor([0.0, 0.0, 0.0])
    expected_std = torch.tensor([0.5, 0.5, 0.5])
    
    assert torch.allclose(mean, expected_mean, atol=0.1), f"Mean should be close to {expected_mean}, but got {mean}"
    assert torch.allclose(std, expected_std, atol=0.1), f"Standard deviation should be close to {expected_std}, but got {std}"

if __name__ == "__main__":
    pytest.main()
