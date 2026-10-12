import torch.nn as nn
import torch.nn.functional as F


class NetworkCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 6, 5)
        self.fc1 = nn.Linear(6 * 53 * 53, 100)
        self.fc2 = nn.Linear(100, 3)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 53 * 53)
        x = F.relu(self.fc1(x))
        return self.sm(self.fc2(x))
