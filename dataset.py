import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import random
from IPython.core.debugger import set_trace
import tifffile as tif
import spectral
from AffineT import *
class IRDataset(Dataset):
    def __init__(self, ir_dir, labels_dir,mode = 'train'):
        self.ir_dir = ir_dir
       
        self.labels_dir = labels_dir
        self.ir_files = os.listdir(ir_dir)
     
        self.labels_files = os.listdir(labels_dir)
        self.mode = mode
    def __len__(self):
        return len(self.ir_files)

    def __getitem__(self, idx):
        ir_file = self.ir_files[idx]
 
        labels_file = self.labels_files[idx]

        ir_data = np.load(os.path.join(self.ir_dir, ir_file))
        ir_data = np.transpose(ir_data, (2, 0, 1))

        ir_data[ir_data<0] = 0

        
        # Affine
        shear = random.uniform(-.35,0.35)
        translation = (random.uniform(-.25,0.25),random.uniform(-.25,0.25))
        rotation = random.uniform(0,180)
        scale = (random.uniform(.65,1.35),random.uniform(.65,1.35))
        affine_transform = Affine(rotation_range=rotation, zoom_range = scale, translation_range=translation)

        

        combined_data = ir_data

        label_data = np.load(os.path.join(self.labels_dir, labels_file))
        
        
        if self.mode == 'train':
            combined_data = (affine_transform(torch.from_numpy(combined_data).float()) - .5)/.5
            label_data = (affine_transform(torch.from_numpy(label_data).unsqueeze(0))).squeeze(0).long()
        else:
            combined_data = ((torch.from_numpy(combined_data).float()) - .5)/.5
            label_data = ((torch.from_numpy(label_data).unsqueeze(0))).squeeze(0).long()
        return combined_data, label_data

