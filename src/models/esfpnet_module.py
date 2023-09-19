from typing import Any, List

import os
import torch
import pyrootutils
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import Dice
import torch.nn.functional as F
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.dc_ce_loss import DC_and_CE_loss
from torch.nn import CrossEntropyLoss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ange_structure_loss(pred, mask, smooth=1):

    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + smooth)/(union - inter + smooth)

    threshold = torch.tensor([0.5]).to(device)
    pred = (pred > threshold).float() * 1

    pred = pred.data.cpu().numpy().squeeze()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    # print(type(pred))
    return (wbce + wiou).mean(), torch.from_numpy(pred)

def dice_loss_coff(pred, target, smooth = 0.0001):

    num = target.size(0)
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)

    return loss.sum()/num

class ESFPModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss: str,
        weight: List[int] = None
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.img_size= [352,352]
        # loss function
        if loss == 'dc-ce':
            self.criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        elif loss == 'ce':
            self.criterion = CrossEntropyLoss(weight=torch.Tensor(weight))
        else:
            raise ValueError('Not implement loss')

        # metric objects for calculating and averaging accuracy across batches
        self.train_dice = Dice().to(device)
        self.val_dice = Dice().to(device)
        self.test_dice = Dice().to(device)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_dice_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        pred = self.net(x)
        pred = F.upsample(pred, size=self.img_size, mode='bilinear', align_corners=False)
        # print(pred.shape)
        return pred

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_dice_best doesn't store accuracy from these checks
        self.val_dice_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        y = y.to(torch.int64)
        logits = self.forward(x)
        loss, preds = ange_structure_loss(logits, y.float())
        # loss = self.criterion(logits, y.squeeze(1))
        # preds = torch.argmax(logits, dim=1)
        return loss, preds.to(device), y.to(device)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        # print(preds, targets)
        self.train_dice(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_dice(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_dice.compute()  # get current val acc
        self.val_dice_best(acc)  # update best so far val acc
        # log `val_dice_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/dice_best", self.val_dice_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_dice(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # _ = UNetPlusPlusModule(None, None, None)
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model")

    @hydra.main(version_base=None, config_path=config_path, config_name="unetplusplus_module.yaml")
    def main(cfg: DictConfig):
        unetplusplus_module = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 1, 240, 240)
        logits = unetplusplus_module(x)
        preds = torch.argmax(logits, dim=1)
        print(logits.shape, preds.shape)
    
    main()