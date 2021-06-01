import torch
from torch import nn

class XYMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        x_loss = self.criterion(input[:, 0], target[:, 0])
        y_loss = self.criterion(input[:, 1], target[:, 1])
        xy_loss = self.criterion(input, target)
        return torch.stack([x_loss, y_loss, xy_loss])
