import torch.nn.functional as F
from torch import nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=[2, 1], reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        eps = torch.finfo().eps
        one_hot_presentation = F.one_hot(targets, 2)
        
        probs = F.softmax(logits, dim=1)
        probs = torch.sum(probs*one_hot_presentation, dim=1)

        loss = -(1 - probs)**self.gamma*torch.log(probs + eps)
        loss = torch.where(targets == 1, self.alpha[1]*loss, self.alpha[0]*loss)
        return self._reduce(loss)
    
    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x

if __name__ == "__main__":
    loss = FocalLoss()
    logits = torch.randn(4, 2)
    targets = torch.randint(0, 2, (4, 1))

    print(logits.shape, targets.shape)
    print(loss(logits, targets))