import torch
import torch.nn as nn
torch.manual_seed(0)

class MLP(nn.Module):
    def __init__(self, dim=64, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))

model = MLP()
x = torch.randn(1, 8, 64)
