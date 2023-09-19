import os 
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms 
from typing import Any, Dict, Optional, Tuple
import time
import csv


class LIDC_IDRI_Dataset(Dataset):
    def __init__(self, nodule_path, clean_path, mode, img_size=[128, 128]):

        # nodule_path: path to dataset nodule image folder
        # clean_path: path to dataset clean image folder
        super().__init__()   
        self.nodule_path = nodule_path
        self.clean_path = clean_path
        self.mode = mode
        self.resize = transforms.Resize(img_size)

        # define function to get list of (image, mask)
        self.file_list = self._get_file_list()

        print(len(self.file_list))

    def __len__(self):
        return len(self.file_list)
    
    def _get_file_list(self):
        file_list = []
        for dicom_path in self.nodule_path:
            # Get mask path of nodule image
            mask_path = dicom_path.replace("Image", "Mask")
            mask_path = mask_path.replace("NI", "MA")

            # Check whether mask path exist
            if os.path.exists(mask_path):

                image = np.load(dicom_path)
                # print(image)

                # image = self._normalize_image(image)
                mask = np.load(mask_path)

                # convert image, mask to tensor

                image = torch.from_numpy(image).to(torch.float)
                mask = torch.from_numpy(mask).to(torch.float)

                # add batch dimension 

                image = image.unsqueeze(0)
                mask = mask.unsqueeze(0)
                file_list.append((image, mask))
        
        for dicom_path in self.clean_path:
            # Get mask path of nodule image
            mask_path = dicom_path.replace("Image", "Mask")
            mask_path = mask_path.replace("CN", "CM")

            # Check whether mask path exist

            if os.path.exists(mask_path):

                image = np.load(dicom_path)
                # print(np.max(image))

                # image = self._normalize_image(image)
                mask = np.load(mask_path)

                # convert image, mask to tensor

                image = torch.from_numpy(image).to(torch.float)
                mask = torch.from_numpy(mask).to(torch.float)

                # add batch dimension 

                image = image.unsqueeze(0)
                mask = mask.unsqueeze(0)

                file_list.append((image, mask))

        return file_list

    def __getitem__(self, index):
        image, mask = self.file_list[index]
        return self.resize(image), self.resize(mask)
    
    def _normalize_image(self, image):
        min_val = np.min(image)
        max_val = np.max(image)

        if max_val - min_val > 0:
            image = (image - min_val) / (max_val - min_val)

        return image
    

class LIDCDataModule(LightningDataModule):
    def __init__(
        self,
        nodule_dir: str = "/work/hpc/iai/loc/LIDC-IDRI-Preprocessing/data/Image",
        clean_dir: str = "/work/hpc/iai/loc/LIDC-IDRI-Preprocessing/data/Clean/Image",
        train_val_test_split: Tuple[int, int, int] = (3, 1, 1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_nodule: int = 1000,
        num_clean: int = 1000,
        img_size=[128, 128],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.nodule_dir = nodule_dir
        self.clean_dir = clean_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory


        # get all file_name in folder

        file_nodule_list = []
        file_clean_list = []
        self.num_nodule = num_nodule
        self.num_clean = num_clean

        # get full path of each nodule file
        for root, _, files in os.walk(self.nodule_dir):
            for file in files:
                if file.endswith(".npy"):
                    dicom_path = os.path.join(root, file)
                    file_nodule_list.append(dicom_path)
        
        # get full path of each clean file
        for root, _, files in os.walk(self.clean_dir):
            for file in files:
                if file.endswith(".npy"):
                    dicom_path = os.path.join(root, file)
                    file_clean_list.append(dicom_path)

        file_nodule_list = file_nodule_list[:self.num_nodule]

        file_clean_list = file_clean_list[:self.num_clean]

        nodule_train, nodule_val, nodule_test = self.split_data(file_nodule_list, train_val_test_split)

        clean_train, clean_val, clean_test = self.split_data(file_clean_list, train_val_test_split)

        self.data_train = LIDC_IDRI_Dataset(nodule_train, clean_train, mode="train", img_size=img_size)

        self.data_val = LIDC_IDRI_Dataset(nodule_val, clean_val, mode="valid", img_size=img_size)

        self.data_test = LIDC_IDRI_Dataset(nodule_test, clean_test, mode="test", img_size=img_size)


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
    datamodule = LIDCDataModule(num_nodule=1000, num_clean=500)
    train_dataloader = datamodule.train_dataloader()
    batch_image = next(iter(train_dataloader))
    images, labels = batch_image