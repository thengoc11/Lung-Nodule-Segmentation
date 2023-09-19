from typing import Any, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import Callback

class WandbCallback(Callback):
    def __init__(self, grid_shape: Tuple[int, int]):
        self.grid_shape = grid_shape
        self.images = {
            'train': None,
            'val': None,
            'test': None,
        }
        self.labels = {
            'train': None,
            'val': None,
            'test': None,
        }
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningDataModule):
        self.callback(trainer, pl_module, mode='train')

    def on_validation_epoch_end(self, trainer: pl.Trainer,
                                pl_module: pl.LightningDataModule):
        self.callback(trainer, pl_module, mode='val')

    def on_test_epoch_end(self, trainer: pl.Trainer,
                          pl_module: pl.LightningDataModule):
        self.callback(trainer, pl_module, mode='test')

    def callback(self, trainer: pl.Trainer, pl_module: pl.LightningDataModule, mode: str):
        with torch.no_grad():
            # print(self.grid_shape[0] * self.grid_shape[1])
            x = self.images[mode][:self.grid_shape[0]]
            y = self.labels[mode][:self.grid_shape[0]]
            
            preds = pl_module.net(x)
        
            preds = F.upsample(preds, size=[352,352], mode='bilinear', align_corners=False)
            preds = preds.sigmoid()
            threshold = torch.tensor([0.5]).to(self.device)
            preds = (preds > threshold).float() * 1
            preds = preds.data.cpu().numpy().squeeze()
            preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
            preds = torch.from_numpy(preds)
            # print(preds.shape)
            x = x.detach().cpu()
            y = y.detach().cpu()

            x = make_grid(x, nrow=self.grid_shape[0])
            y = make_grid(y, nrow=self.grid_shape[0])
            preds = make_grid(preds, nrow=self.grid_shape[0])

            logger = trainer.logger 
            logger.log_image(key=mode+'/inference', images=[x, y, preds], caption=["image", "grow-truth", "predict"])

            self.images[mode] = None
            self.labels[mode] = None
            
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
       self.store_data(batch, mode='train')

    def on_validation_batch_end(self, trainer: pl.Trainer,
                                pl_module: pl.LightningModule, outputs: Any,
                                batch: Any, batch_idx: int,
                                dataloader_idx: int) -> None:
        self.store_data(batch, mode='val')

    def on_test_batch_end(self, trainer: pl.Trainer,
                          pl_module: pl.LightningModule, outputs: Any,
                          batch: Any, batch_idx: int,
                          dataloader_idx: int) -> None:
        self.store_data(batch, mode='test')

    def store_data(self, batch: Any, mode: str):
        if self.images[mode] == None:
            self.images[mode], self.labels[mode] = batch
        elif self.images[mode].shape[0] < self.grid_shape[0] * self.grid_shape[1]:
            self.images[mode] = torch.cat((self.images[mode], batch[0]), dim=0)
            self.labels[mode] = torch.cat((self.labels[mode], batch[1]), dim=0)
