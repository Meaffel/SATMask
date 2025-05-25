from torch import nn
from torch.nn import functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 1) Pool 28×28 -> 7×7
        #    Since 28/4 = 7 exactly, we can do
        self.pool = nn.AvgPool2d(kernel_size=7, stride=7)

        # Alternatively, for arbitrary input you can do
        # self.pool = nn.AdaptiveAvgPool2d((7,7))

        # 2) Single conv: 1×7×7 -> C×7×7
        #    Here I keep C=1, but you can raise to e.g. 16 if you like.
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        # 3) FC from (1×7×7)=49 features -> n_classes
        self.fc   = nn.Linear(1 * 4 * 4, 1)

    def forward(self, x):
        # x: (B,1,28,28)
        x = self.pool(x)                # -> (B,1,7,7)
        x = F.relu(self.conv(x))        # -> (B,1,7,7)
        x = x.view(x.size(0), -1)       # -> (B, 49)
        x = self.fc(x)                  # -> (B, n_classes)
        return x