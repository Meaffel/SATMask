import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN_28x28_k7, SimpleCNN_56x56_k14, SimpleCNN_84x84_k21, SimpleCNN_112x112_k28

# 1. Hyperparameters
batch_size   = 64
learning_rate= 5e-3
num_epochs   = 10
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

keep = (train_dataset.targets == 0) | (train_dataset.targets == 1)
train_dataset.data    = train_dataset.data[keep]
train_dataset.targets = train_dataset.targets[keep]

keep = (test_dataset.targets == 0) | (test_dataset.targets == 1)
test_dataset.data    = test_dataset.data[keep]
test_dataset.targets = test_dataset.targets[keep]

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2,
                          pin_memory=True)
test_loader  = DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=2,
                          pin_memory=True)

# 3. Define the CNN (no MaxPool; use stride=2 in convs)
for HW in [112]:
    if HW == 28:
        model = SimpleCNN_28x28_k7().to(device)
    elif HW == 56:
        model = SimpleCNN_56x56_k14().to(device)
    elif HW == 84:
        model = SimpleCNN_84x84_k21().to(device)
    elif HW == 112:
        model = SimpleCNN_112x112_k28().to(device)

    # 4. Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Training loop
    def train():
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data = data.to(device)                     # (B,1,32,32)
            data = F.interpolate(data, size=(HW, HW), mode='bilinear', align_corners=False)
            # 2) reshape & cast target -> float tensor (B,1)
            target = target.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            output = model(data)                       # (B,1) logits
            loss   = criterion(output, target)         
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                avg = running_loss / 100
                print(f"Train Batch {batch_idx}/{len(train_loader)}  Loss: {avg:.4f}")
                running_loss = 0.0

    # 6. Evaluation loop
    def test():
        model.eval()
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                data = F.interpolate(data, size=(HW, HW), mode='bilinear', align_corners=False)
                target = target.to(device).unsqueeze(1).float()

                output = model(data)                   # (B,1)
                test_loss += criterion(output, target).item()

                # 3) apply sigmoid + threshold at 0.5
                preds = (torch.sigmoid(output) > 0.5).long()  # (B,1)
                correct += preds.eq(target.long()).sum().item()

        test_loss /= len(test_loader)
        acc = 100. * correct / len(test_dataset)
        print(f"Test Loss: {test_loss:.4f},  Acc: {correct}/{len(test_dataset)} ({acc:.2f}%)\n")

    # 7. Run
    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}:')
        train()
        test()

    # 8. Save
    if HW == 28:
        torch.save(model.state_dict(), 'mnist_cnn_k7.pth')
        print("Saved model to mnist_cnn_k7.pth")
    elif HW == 56:
        torch.save(model.state_dict(), 'mnist_cnn_k14.pth')
        print("Saved model to mnist_cnn_k14.pth")
    elif HW == 84:
        torch.save(model.state_dict(), 'mnist_cnn_k21.pth')
        print("Saved model to mnist_cnn_k21.pth")
    elif HW == 112:
        torch.save(model.state_dict(), 'mnist_cnn_k28.pth')
        print("Saved model to mnist_cnn_k28.pth")
