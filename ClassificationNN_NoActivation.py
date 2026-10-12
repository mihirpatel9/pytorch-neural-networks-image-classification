"""Run the assignment's no-activation study: --hidden 5 or --hidden 30."""
import argparse
import torch
import Utils
from NetworkNoActivation import Network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, choices=[5, 30], default=5)
    args = parser.parse_args()
    device = torch.device('cpu')
    loader = Utils.get_data_loader(batch_size=1)
    model = Network(args.hidden).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.MSELoss()
    model.train()
    for epoch in range(500):
        epoch_loss = 0.0
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 25 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}')
    model.eval()
    eval_loader = Utils.get_data_loader(batch_size=200)
    x, y = next(iter(eval_loader))
    x = x.to(device, dtype=torch.float32)
    y = y.to(device, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        truth = y.argmax(dim=1)
        accuracy = (pred == truth).float().mean().item() * 100
    print(f'Hidden neurons: {args.hidden}; Accuracy: {accuracy:.2f}%')
    Utils.plot_decision_boundary(model, x.cpu().numpy(), truth.cpu().numpy())


if __name__ == '__main__':
    main()
