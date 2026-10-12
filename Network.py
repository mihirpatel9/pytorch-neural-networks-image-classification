import torch


class Network(torch.nn.Module):
    def __init__(self, hidden_size, output_size=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.act = torch.nn.ReLU()
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.sm(self.fc2(x))
        return x
