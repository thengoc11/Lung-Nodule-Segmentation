
# LUNG NODULE SEGMENTATION
Welcome to the LUNG NODULE SEGMENTATION repository. This project provides an easy-to-use framework of the segmentation model for Medical Image, allowing you to effectively train/evaluate.

## Requirements
All the dependencies can be installed using the provided requirements.txt file.
## Installation
1. Clone the repository:
   ```
   git clone https://github.com/thengoc11/Lung-Nodule-Segmentation
   ```
2. Change the directory:
   ```
   cd Lung-Nodule-Segmentation
   ```
3. Create a conda environment and install dependencies:
   ```
   conda create -n lung python=3.10
   ```
4. Activate the conda environment:
   ```
   conda activate lung
   ```
## Dataset
1. Change the directory:
   ```
   mkdir data
   cd data
   ```
2. Download LIDC Dataset:
   [LIDC data](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254&fbclid=IwAR1vDkrpq0IJN8KwPT2Fft1GJ4bFPiMqXp4p08eEfOaUYofS-88pnNF_Z7g)
4. Data Preprocessing with repository
   ```
   git clone https://github.com/jaeho3690/LIDC-IDRI-Preprocessing
   ```
   
## How to run

#### Train model with default configuration

```
# before training, set env WANDB_API_KEY to log with wandb logger
export WANDB_API_KEY = ${WANDB_API_KEY}

# train on one GPU
python src/train.py trainer=gpu logger=wandb

# train on multi GPU
python src/train.py trainer=ddp trainer.devices=4 logger=wandb
```
#### Train with other models

1. Customize dataloader for your dataset:
   ```bash
    File path: ./src/data/{your_datamodule}.py
   ```
2. Add your model architecure:
   ```bash
    File path: ./src/models/{your_module}.py
   ```
3. Edit visualization with Wandb:
   ```bash
    File path: ./src/utils/callbacks/wandb_callback.py
   ```
4. Modify config dataset:
   ```bash
    ./configs/data/{your_dataset}.yaml
   ```
   Modify config training:
      ```bash
      ./configs/train.yaml
      ```
5. Training
    Train on single GPU:
    ```bash
    python src/train.py trainer=gpu logger=wandb
    ```
    
    Train on multi GPU:
    ```bash
    python src/train.py trainer=ddp trainer.devices=4 logger=wandb
    ```
        



