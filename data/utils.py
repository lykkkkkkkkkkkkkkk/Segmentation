import random
import torch
from PIL import Image, ImageEnhance, ImageOps
import numpy as np


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image) -> (Image.Image, Image.Image):
        if random.random() < self.p:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        return image, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image) -> (Image.Image, Image.Image):
        if random.random() < self.p:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
        return image, mask


class RandomRotation:
    def __init__(self, degrees: float = 30.0):
        self.degrees = degrees

    def __call__(self, image: Image.Image, mask: Image.Image) -> (Image.Image, Image.Image):
        angle = random.uniform(-self.degrees, self.degrees)
        image = image.rotate(angle, resample=Image.BILINEAR)
        mask = mask.rotate(angle, resample=Image.NEAREST)
        return image, mask


class RandomResizedCrop:
    def __init__(self, size: int = 256, scale: tuple = (0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, image: Image.Image, mask: Image.Image) -> (Image.Image, Image.Image):
        width, height = image.size
        area = width * height
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        new_width = int(round(np.sqrt(target_area * aspect_ratio)))
        new_height = int(round(np.sqrt(target_area / aspect_ratio)))

        if new_width > width:
            new_width = width
        if new_height > height:
            new_height = height

        x1 = random.randint(0, width - new_width)
        y1 = random.randint(0, height - new_height)

        image = image.crop((x1, y1, x1 + new_width, y1 + new_height))
        mask = mask.crop((x1, y1, x1 + new_width, y1 + new_height))

        image = image.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        return image, mask


class RandomBrightness:
    def __init__(self, brightness: float = 0.5):
        self.brightness = brightness

    def __call__(self, image: Image.Image, mask: Image.Image) -> (Image.Image, Image.Image):
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        image = enhancer.enhance(factor)
        return image, mask


class RandomContrast:
    def __init__(self, contrast: float = 0.5):
        self.contrast = contrast

    def __call__(self, image: Image.Image, mask: Image.Image) -> (Image.Image, Image.Image):
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        image = enhancer.enhance(factor)
        return image, mask


class RandomSaturation:
    def __init__(self, saturation: float = 0.5):
        self.saturation = saturation

    def __call__(self, image: Image.Image, mask: Image.Image) -> (Image.Image, Image.Image):
        enhancer = ImageEnhance.Color(image)
        factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        image = enhancer.enhance(factor)
        return image, mask


class RandomHue:
    def __init__(self, hue: float = 0.1):
        self.hue = hue

    def __call__(self, image: Image.Image, mask: Image.Image) -> (Image.Image, Image.Image):
        hue = random.uniform(-self.hue, self.hue)
        image = ImageOps.colorize(image.convert('L'), (0, 0, 0), (hue, hue, hue))
        return image, mask


class ToTensor:
    def __call__(self, image: Image.Image, mask: Image.Image) -> (torch.Tensor, torch.Tensor):
        image = torch.tensor(np.array(image)).float().permute(2, 0, 1) / 255.0
        mask = torch.tensor(np.array(mask)).long()
        return image, mask


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return image, mask


class Composed:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image) -> (torch.Tensor, torch.Tensor):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask
