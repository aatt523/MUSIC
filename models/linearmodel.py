import torch.nn as nn
import torch.nn.functional as F
class linear_model(nn.Module):
    def __init__(self, output_dim=5, input_dim=512):
        super(linear_model, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        out = F.normalize(x, p=2.0)
        out = self.linear(out)
        return out
