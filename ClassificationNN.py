import sys
import Utils
import torch
from Network import Network


def main():
    device = torch.device('cpu')  # Efficient for this tiny batch-size-1 model
    data_loader = Utils.get_data_loader(batch_size=1)
    loss_fn = torch.nn.MSELoss()
    hidden_size = 5  # Experiments: 1, 3, 5, 30
    net = Network(hidden_size).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    num_epochs = 500

    net.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for _, (x, y) in enumerate(data_loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = net(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 25 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {running_loss:.4f}')

    net.eval()
    test_loader = Utils.get_data_loader(batch_size=200)
    x, y = next(iter(test_loader))
    x = x.to(device, dtype=torch.float32)
    y = y.to(device, dtype=torch.float32)
    with torch.no_grad():
        outputs = net(x)
        predicted = torch.argmax(outputs, dim=1)
        actual = torch.argmax(y, dim=1)
        accuracy = (predicted == actual).float().mean().item()
    print(f'Accuracy: {accuracy * 100:.2f}%')
    Utils.plot_decision_boundary(net, x.cpu().numpy(), actual.cpu().numpy())


if __name__ == '__main__':
    sys.exit(int(main() or 0))
