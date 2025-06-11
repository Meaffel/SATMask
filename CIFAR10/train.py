import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from model import SimpleCNN

# 1) Hyper-parameters & device
batch_size    = 64
learning_rate = 5e-3
num_epochs    = 10
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Compute gray-scale mean/std by averaging CIFAR-10 RGB stats
CIFAR10_MEAN_RGB = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_RGB  = (0.2023, 0.1994, 0.2010)
mean_gray = sum(CIFAR10_MEAN_RGB) / 3.0    # 0.4733
std_gray  = sum(CIFAR10_STD_RGB ) / 3.0    # 0.2009

# 3) Transforms: only Grayscale -> ToTensor -> Normalize
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((mean_gray,), (std_gray,)),
])
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((mean_gray,), (std_gray,)),
])

# 4) Load CIFAR-10
train_dataset = datasets.CIFAR10('./data', train=True,  download=True, transform=train_transform)
test_dataset  = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

# 5) Filter to boats(ship)=8 vs frogs=6, remap ship->0, frog->1
def filter_boats_frogs(dataset):
    data    = dataset.data                            # numpy array (N,32,32,3)
    targets = np.array(dataset.targets)                # (N,)
    mask    = (targets == 6) | (targets == 8)
    dataset.data    = data[mask]
    dataset.targets = (targets[mask] == 6).astype(np.int64).tolist()

filter_boats_frogs(train_dataset)
filter_boats_frogs(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

model = SimpleCNN().to(device)

# 1) Use BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 7) Training loop
def train():
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data = data.to(device)                     # (B,1,32,32)
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

# 8) Evaluation loop
def test():
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device).unsqueeze(1).float()

            output = model(data)                   # (B,1)
            test_loss += criterion(output, target).item()

            # 3) apply sigmoid + threshold at 0.5
            preds = (torch.sigmoid(output) > 0.5).long()  # (B,1)
            correct += preds.eq(target.long()).sum().item()

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_dataset)
    print(f"Test Loss: {test_loss:.4f},  Acc: {correct}/{len(test_dataset)} ({acc:.2f}%)\n")


# 9) Run training & testing
for epoch in range(1, num_epochs+1):
    print(f"Epoch {epoch}/{num_epochs}")
    train()
    test()

# 10) Save model
torch.save(model.state_dict(), 'cifar10_gray_boat_vs_frog.pth')
print("Saved model to cifar10_gray_boat_vs_frog.pth")