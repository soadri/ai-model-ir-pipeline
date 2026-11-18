import torch
import torch.nn as nn
torch.manual_seed(0)

class SwiGLU(nn.Module):
    def __init__(self, dim=64, hidden=128):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

model = SwiGLU()
x = torch.randn(1, 8, 64)
