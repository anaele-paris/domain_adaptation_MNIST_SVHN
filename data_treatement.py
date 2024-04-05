import albumentations as A
import torch
from torch.utils.data import Dataset

def generate_rotation(image, param=20, number=16):
    aug_img = []
    rotate = A.Compose([A.Rotate(limit=param, p=1)])

    for _ in range(number):
        to_add = torch.from_numpy(rotate(image=image.numpy())["image"])
        aug_img.append(to_add)

    return aug_img

def generate_gaussian_blur(image, param=0.5, number=16):
    aug_img = []
    gaussian_blur = A.Compose([A.GaussianBlur(p=param)])

    for _ in range(number):
        to_add = torch.from_numpy(gaussian_blur(image=image.numpy())["image"])
        aug_img.append(to_add)

    return aug_img

def generate_brightness_contrast(image, param=0.5, number=16):
    aug_img = []
    brightness_contrast = A.Compose([A.RandomBrightnessContrast(p=param)])

    for _ in range(number):
        to_add = torch.from_numpy(brightness_contrast(image=image.numpy())["image"])
        aug_img.append(to_add)

    return aug_img

def increase_data(dataset, number=10):
    all_samples = []
    all_labels = []

    for image, label in dataset:
        rotation_images = generate_rotation(image, number=number)
        gaussian_blur_images = generate_gaussian_blur(image, number=number)
        brightness_contrast_images = generate_brightness_contrast(image, number=number)

        all_samples.append(image)
        all_samples.extend(rotation_images)
        all_samples.extend(gaussian_blur_images)
        all_samples.extend(brightness_contrast_images)

        all_labels.append(label)
        all_labels.extend([label] * len(rotation_images))
        all_labels.extend([label] * len(gaussian_blur_images))
        all_labels.extend([label] * len(brightness_contrast_images))

    return all_samples, all_labels

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label

def increase_dataset(dataset):
    new_data, new_labels = increase_data(dataset, 5)
    new_dataset = CustomDataset(data=new_data, labels=new_labels)
    return new_dataset