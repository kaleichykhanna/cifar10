import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

class CIFAR10Dataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(labels_file, index_col='id')
        self.labels = pd.get_dummies(self.labels)  # Convert labels to dummy variables
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, f"{idx+1}.png")
        image = Image.open(img_name)
        label = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float32)  # Convert label to tensor

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def create_dataloaders(image_dir, labels_file, batch_size, transform=transform):
    dataset = CIFAR10Dataset(image_dir, labels_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dataloader
