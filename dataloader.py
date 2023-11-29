from torch.utils import data
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import os
import argparse

def create_dataset(args):
    print("Build dataset...")

    train_dataset = CelebA_HQ(root=args.root_path, mode="train", scale=args.scale)
    val_datset = CelebA_HQ(root=args.root_path, mode="val", scale=args.scale)
    test_dataset  = CelebA_HQ(root=args.root_path, mode="test", scale=args.scale)

    ## Create batch dataset
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = data.DataLoader(dataset=val_datset, batch_size=args.bs, shuffle=True)
    test_loader  = data.DataLoader(dataset=test_dataset , batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader

def getData(root, mode):
    folder_path = os.path.join(root, mode, 'hq')

    # Get a list of all files in the folder and its subdirectories
    filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return filenames

class CelebA_HQ(data.Dataset):
    def __init__(self, root, mode, scale):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name = getData(self.root, mode)
        self.mode = mode
        self.scale = scale
        print("> Found %d images..." % (len(self.img_name)))


    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):

        lq_path = f"{self.root}/{self.mode}/lq/{self.scale}/{self.img_name[index]}.jpg"
        hq_path = f"{self.root}/{self.mode}/hq/{self.img_name[index]}.jpg"
        kernel_path = f"{self.root}/{self.mode}/kernel/{self.img_name[index]}.npy"

        kernel = torch.from_numpy(np.load(kernel_path)).float().unsqueeze(0)

                
        ## Define transformations
        transformations = transforms.Compose([
            # transforms.Resize([256, 256]),
            transforms.ToTensor(), ## Scales the pixel to the range [0, 1]
            # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)) ## Normalization
        ])

        lq = Image.open(lq_path).convert("RGB")
        lq = transformations(lq)
        hq = Image.open(hq_path).convert("RGB")
        hq = transformations(hq)

        return hq, lq, kernel
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

	## Mode
    parser.add_argument("--bs", type=int, default=4, help="Batch size during training")
    parser.add_argument("--scale", type=int, default=4, help="sr scale")
	## Paths
    parser.add_argument("--root_path", type=str, default="/eva_data2/shlu2240/Dataset/celeba_hq")

    args = parser.parse_args()
    train_loader, val_loader, test_loader = create_dataset(args)

    for img, kernel in val_loader:
        print(img.shape)
        print(kernel.shape)