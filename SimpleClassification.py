import sys
import argparse
from pathlib import Path
import torch
from torchvision import transforms
import UtilsACS
from NetworkLinear import NetworkLinear
from NetworkCNN import NetworkCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['linear', 'cnn'], default='linear')
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('Using device:', device)

    data_dir = Path(__file__).resolve().parent / 'data' / 'Dataset_PlanesCarsShips'
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    batch_size = 16
    train_loader = UtilsACS.get_train_loader(str(data_dir / 'train'), batch_size, image_transform)
    test_loader = UtilsACS.get_test_loader(str(data_dir / 'test'), batch_size, image_transform)
    print('Training batches:', len(train_loader))
    print('Testing batches:', len(test_loader))

    if args.model == 'linear':
        net = NetworkLinear()
        print('Model: Linear Neural Network')
    else:
        net = NetworkCNN()
        print('Model: Convolutional Neural Network')
    net = net.to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print('\nStarting training...\n')

    for epoch in range(args.epochs):
        net.train()
        loss_sum = 0.0
        correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs.clamp_min(1e-8).log(), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            pred = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        print(f'Epoch [{epoch + 1}/{args.epochs}] '
              f'Loss: {loss_sum / total:.4f} Accuracy: {100 * correct / total:.2f}%',
              flush=True)

    net.eval()
    total = correct = 0
    print('\nEvaluating model...')
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            pred = net(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    print('------------------------------')
    print('Model:', args.model)
    print('Total Test Images:', total)
    print('Correct Predictions:', correct)
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    print('------------------------------')
    name = f'model_{args.model}.pth'
    torch.save(net.state_dict(), name)
    print('Model saved:', name)


if __name__ == '__main__':
    sys.exit(int(main() or 0))
