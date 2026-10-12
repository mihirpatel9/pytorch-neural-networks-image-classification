import os
import torch
from PIL import Image


class ACSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.all_image_paths = []
        self.all_labels = []
        for category, label in {'airplanes': 0, 'cars': 1, 'ships': 2}.items():
            category_path = os.path.join(data_dir, category)
            if not os.path.isdir(category_path):
                raise FileNotFoundError(f'Dataset folder not found: {category_path}')
            paths = [os.path.join(category_path, f)
                     for f in sorted(os.listdir(category_path))
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.all_image_paths.extend(paths)
            self.all_labels.extend([label] * len(paths))
        self.num_classes = 3

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, index):
        image_path = self.all_image_paths[index]
        label = self.all_labels[index]
        with Image.open(image_path) as image:
            img = image.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
