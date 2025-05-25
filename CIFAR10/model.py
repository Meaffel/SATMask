from torch import nn
from torch.nn import functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, base_ch=1):
        super(SimpleCNN, self).__init__()
        # 1) Pool 32×32 -> 8×8
        #    32/4 = 8 exactly, so kernel_size=4,stride=4 works
        self.pool = nn.AvgPool2d(kernel_size=8, stride=8)

        # 2) Conv: in_channels=3 -> base_ch=1, keeps 8×8 spatial
        self.conv = nn.Conv2d(in_channels,
                              base_ch,
                              kernel_size=3,
                              stride=1,
                              padding=1)

        # 3) FC from 16×8×8 -> n_classes
        self.fc = nn.Linear(base_ch * 4 * 4, 1)

    def forward(self, x):
        # x: (B,3,32,32)
        x = self.pool(x)              
        x = F.relu(self.conv(x))      
        x = x.view(x.size(0), -1)     
        x = self.fc(x)                # -> (B, n_classes)
        return x