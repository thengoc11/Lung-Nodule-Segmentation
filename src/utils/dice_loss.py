import torch.nn.functional as F
from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps=eps

    def forward(self, logits, targets):
        # logits: n, 1, w, h
        # targets: n, 1, w, h

        loss = 2 * logits * targets
        loss = (loss.sum(dim=[1,2,3]) + self.eps) / (logits.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3]) + self.eps)

        loss = -torch.log(loss)

        return loss.mean()

if __name__ == "__main__":
    loss = DiceLoss()
    logits = torch.rand(4, 1, 10, 10)
    targets = torch.randint(0, 2, (4, 1, 10, 10))

    print(logits.shape, targets.shape)
    print(logits.type(), targets.type())
    print(loss(logits, targets))