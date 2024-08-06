import pandas as pd
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

# Custom pytorch Dataset function
class PcPartsData(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir  
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(image_path).float()
        image = transforms.ToPILImage()(image) 
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
