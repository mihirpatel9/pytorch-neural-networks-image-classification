import torch


class Network(torch.nn.Module):
    """Two affine layers without intermediate or output activations."""
    def __init__(self, hidden_size, output_size=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
