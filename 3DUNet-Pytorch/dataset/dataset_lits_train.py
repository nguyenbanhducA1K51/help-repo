from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset

from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize
# from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize

class Train_Dataset(dataset):
    def __init__(self, args):

        self.args = args

        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))

        self.transforms = Compose([
            #args.crop_size=48
                RandomCrop(self.args.crop_size),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                # RandomRotate()
            ])

    def __getitem__(self, index):

        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        # print ("ct1",ct_array.shape)
        seg_array = sitk.GetArrayFromImage(seg)
# norm_factor' default=200.0)
        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)
        # print ("ct2",ct_array.shape)
        if self.transforms:
            ct_array,seg_array = self.transforms(ct_array, seg_array)     
        # print (ct_array.shape)
        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
                # in function write_name_list, file preprocess_LiTs : 
                #   f.write(ct_path + ' ' + seg_path + "\n")
        return file_name_list

if __name__ == "__main__":
    sys.path.append('/root/repo/3DUNet-Pytorch/dataset/')
    sys.path.append("/root/repo/3DUNet-Pytorch/")
    from config import args
    train_ds = Train_Dataset(args)
    (ct,seg)=train_ds.__getitem__(0)
    (ct,seg)=train_ds.__getitem__(5)

    train_dl = DataLoader(train_ds, 2, False, num_workers=1)
    # what is false here: false is shuffle
    # for i, (ct, seg) in enumerate(train_dl):
    #     print(i,ct.size(),seg.size())
    # it=iter(train_dl)
    # (ct,seg)=it.next()  
    # size of image : 2, 1, 48, 256, 256]
    # so batch size is 2