import numpy as np
import matplotlib.pyplot as plt
import torch
from MyDataSet import MyDataSet


def visualize_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
    return X, y


def get_data_loader(batch_size=4):
    dataset = MyDataSet()
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def plot_decision_boundary(net, X, y):
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2.5, 2.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xdata = np.c_[xx.ravel(), yy.ravel()]
    device = next(net.parameters()).device
    with torch.no_grad():
        inputs = torch.tensor(xdata, dtype=torch.float32).to(device)
        Z = net(inputs).cpu().numpy()
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Two-moons decision boundary')
    plt.show()
