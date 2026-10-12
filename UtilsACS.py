import torch
import torchvision
import matplotlib.pyplot as plt
from ACSDataSet import ACSDataset


def get_train_loader(data_dir, batch_size, transform=None):
    dataset = ACSDataset(data_dir, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=True, num_workers=0,
                                       pin_memory=False, drop_last=True)


def get_test_loader(data_dir, batch_size, transform=None):
    dataset = ACSDataset(data_dir, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=False, num_workers=0,
                                       pin_memory=False)


def plot_images(images, labels):
    img_grid = torchvision.utils.make_grid(images.cpu(), nrow=4, normalize=True)
    np_img = img_grid.numpy().transpose(1, 2, 0)
    plt.imshow(np_img)
    plt.axis('off')
    plt.show()
