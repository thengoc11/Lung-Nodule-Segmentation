import nibabel as nib
import numpy as np
import os 

# define folder path
folder_path = "/work/hpc/iai/du/self_learning/data/Task01_BrainTumour/labelsTr"
folder_des = "/work/hpc/iai/du/self_learning/data/Task01_BrainTumour/labelsTr_preprocessed"

# list all files in folder
files = os.listdir(folder_path)
# remove all files that have . in the head of file name
files = [file for file in files if file[0] != "."]
files = sorted(files)
print("number of files: ", len(files))

# create folder to save preprocessed data if not exist
if not os.path.exists(folder_des):
    os.makedirs(folder_des)

# loop through all files
for file in files:
    # load nii file
    img_numpy = nib.load(os.path.join(folder_path, file)).get_fdata()
    # shape (240, 240, 155, 4) or (240, 240, 155)
    num_slices = img_numpy.shape[2]
    # reshape to (155, 240, 240, 4) or (155, 240, 240)
    if len(img_numpy.shape) == 4:
        img_numpy = np.transpose(img_numpy, (2, 0, 1, 3))
    else:
        img_numpy = np.transpose(img_numpy, (2, 0, 1))
    print(file, img_numpy.shape)
    # split into 155 slices
    for i in range(num_slices):
        # save each slice as npy file
        np.save(os.path.join(folder_des, file[:-7] + "_" + str(i) + ".npy"), img_numpy[i])
        print("saved: ", os.path.join(folder_des, file[:-7] + "_" + str(i) + ".npy"))
        

