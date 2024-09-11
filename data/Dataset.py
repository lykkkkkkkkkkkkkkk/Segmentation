import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

class TextlineDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.masks = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]
        self.image_base_names = [os.path.splitext(f)[0] for f in self.images]
        self.mask_base_names = [os.path.splitext(f)[0] for f in self.masks]

        self.common_base_names = list(set(self.image_base_names) & set(self.mask_base_names))

        if len(self.common_base_names) == 0:
            raise ValueError("No matching images and masks found.")

        self.images = sorted([os.path.join(image_dir, f + '.jpg') for f in self.common_base_names])
        self.masks = sorted([os.path.join(mask_dir, f + '.png') for f in self.common_base_names])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image and mask
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask


def get_dataloaders(image_dir, mask_dir, transforms, batch_size=8, shuffle=True, num_workers=4, pin_memory=True):

    dataset = TextlineDataset(image_dir=image_dir, mask_dir=mask_dir, transforms=transforms)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader