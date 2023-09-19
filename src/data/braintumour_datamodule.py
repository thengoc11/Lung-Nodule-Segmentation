from typing import Any, Dict, Optional, Tuple
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms 
import glob
from torch import Tensor
import os
import numpy as np

class BrainTumourDataset(Dataset):

    def __init__(self, file_paths, mode= "train", img_size = [128, 128]):
        super().__init__()        
        self.mode = mode
        self.img_paths = file_paths
        
        # data transformations
        self.color_jitter_transform = transforms.ColorJitter(brightness=0.2)
        self.elastic_transform = transforms.ElasticTransform(alpha=720.0, sigma=24.0)
        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize(img_size)
        self.img_size = img_size
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        
        img_path = self.img_paths[index]
        label_path = img_path.replace('images', 'labels')

        img, label = np.load(img_path), np.load(label_path)
        # get only channel 1 of image
        img = img[:, :, 1]

        img = self.totensor(img).to(torch.float)
        img = self.resize(img)

        label = self.totensor(label).to(torch.float)
        label = self.resize(label)

        if self.mode == "train":
            img, label = self.transfrom_train(img, label)

        # img = torch.rand(1, self.img_size[0], self.img_size[1])
        # label = torch.rand(1, self.img_size[0], self.img_size[1])
        
        return img, label
    
    def transfrom_train(self, image, mask):
        img_w, img_h = image.shape[-2:]
        
        # HFLIP
        if np.random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        # VFLIP
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            
        # AFFINE
        affine_param = transforms.RandomAffine.get_params(
        degrees = [-20, 20], translate = [0.1, 0.1],  
        img_size = [img_w, img_h], scale_ranges = [0.9, 1.1], 
        shears = [-0.2, 0.2])
        image = TF.affine(image, 
                        affine_param[0], affine_param[1],
                        affine_param[2], affine_param[3])
        mask = TF.affine(mask, 
                        affine_param[0], affine_param[1],
                        affine_param[2], affine_param[3])
        
        #  ElasticTransform 
        # image = self.elastic_transform(image)
        # mask = self.elastic_transform(mask)
        
        # ColorJitter
        image = self.color_jitter_transform(image)
        
        return image, mask



class BrainTumourDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/Task01_BrainTumour/imagesTr_preprocessed",
        train_val_test_split: Tuple[int, int, int] = (3, 1, 1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_images: int = 1000,
        img_size = [128, 128],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        # get all file_name in folder
        file_names = os.listdir(data_dir)
        # get full path of each file
        file_paths = [os.path.join(data_dir, file_name) for file_name in file_names]

        file_paths = file_paths[:num_images]
        train_paths, val_paths, test_paths = self.split_data(file_paths, train_val_test_split)
        
        self.data_train = BrainTumourDataset(train_paths, mode= "train", img_size=img_size)
        self.data_val = BrainTumourDataset(val_paths, mode= "valid", img_size=img_size)
        self.data_test = BrainTumourDataset(test_paths, mode= "test", img_size=img_size)


    def split_data(self, file_paths, train_val_test_split):
        # get len files
        num_files = len(file_paths)
        
        # ratio
        train_ratio, val_ratio, test_ratio = train_val_test_split
        
        # get num train, val, test
        num_train = int(num_files * train_ratio / (train_ratio + val_ratio + test_ratio))
        num_val = int(num_files * val_ratio / (train_ratio + val_ratio + test_ratio))
        
        # get random index
        train_paths = list(np.random.choice(file_paths, num_train, replace=False))
        val_paths = list(np.random.choice(list(set(file_paths) - set(train_paths)), num_val, replace=False))
        test_paths = list(set(file_paths) - set(train_paths) - set(val_paths))
        return train_paths, val_paths, test_paths
        
    
    @property
    def num_classes(self):
        return 4

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    datamodule = BrainTumourDataModule()
    train_dataloader = datamodule.train_dataloader()
    batch_image = next(iter(train_dataloader))
    images, labels = batch_image

    image = images[:2]
    label = labels[:2]

    print(image.shape, label.shape)    
    print(image.max(), image.min())
    print(label.max(), label.min())
