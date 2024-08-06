import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from get_data import PcPartsData

# Split data into training, validation and test sets
def split_data(dataset, train_ratio=0.7, validation_ratio=0.15):
    train_size = int(train_ratio * len(dataset))
    validation_size = int(validation_ratio * len(dataset))
    test_size = len(dataset) - train_size - validation_size

    train_data, validation_data, test_data = random_split(dataset, [train_size, validation_size, test_size])
    return train_data, validation_data, test_data

# Create Data loaders
def create_dataloaders(train_data, validation_data, batch_size=16):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    return train_loader, validation_loader


# Data Augmentation
def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    return train_transform, val_transform
