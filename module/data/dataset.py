import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

class BirdDataset(Dataset):
    def __init__(self, df, transforms, mode="train"):
        super(BirdDataset, self).__init__()
        self.df = df[df["data set"] == mode].reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path =  "input/" + self.df["filepaths"][idx]
        image = Image.open(img_path)
        image = self.transforms(image=np.array(image))["image"]

        label = self.df["labels"][idx]

        return image, label