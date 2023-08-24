from torch._C import dtype
import sys
sys.path.append("../")
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import math
import SimpleITK as sitk
import config
class Img_DataSet(Dataset):
    def __init__(self, data_path, label_path, args):
        # test_cut_size',  default=48, help='size of sliding window')
        #test_cut_stride' default=24, help='stride of sliding window')
        # meaning of cut_size and cut_stride
        # in config, it said size of sliding window')
        # stride of sliding window')
        self.n_labels = args.n_labels
        self.cut_size = args.test_cut_size
        self.cut_stride = args.test_cut_stride

        # 读取一个data文件并归一化 、resize
        self.ct = sitk.ReadImage(data_path,sitk.sitkInt16)
        self.data_np = sitk.GetArrayFromImage(self.ct)
        self.ori_shape = self.data_np.shape
        # ori is (75, 512, 512)
        self.data_np = ndimage.zoom(self.data_np, (args.slice_down_scale, args.xy_down_scale, args.xy_down_scale), order=3) # 双三次重采样
       #("data shape",self.data_np.shape)  (75,256,256)

        # whhy zoom like this while in preprocess_LiTs, 
        # ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale), order=3)
        self.data_np[self.data_np > args.upper] = args.upper
        self.data_np[self.data_np < args.lower] = args.lower
        self.data_np = self.data_np/args.norm_factor
        self.resized_shape = self.data_np.shape
        # 扩展一定数量的slices，以保证卷积下采样合理运算
        #扩展一定数量的slices，以保证卷积下采样合理运算
        # "Perform patch-based operation 
        # on the data with a certain stride to prevent memory overflow."
        self.data_np = self.padding_img(self.data_np, self.cut_size,self.cut_stride)
        #data_np has shape ( s,h,w),e.g, 96,256,256
        self.padding_shape = self.data_np.shape
        # 对数据按步长进行分patch操作，以防止显存溢出

        self.data_np = self.extract_ordered_overlap(self.data_np, self.cut_size, self.cut_stride)
        # now data_np has shape (patch, s,h,w)
        # in this config, s=48,patch=3
        # 读取一个label文件 shape:[s,h,w]
        self.seg = sitk.ReadImage(label_path,sitk.sitkInt8)
        self.label_np = sitk.GetArrayFromImage(self.seg)
        # purpose of this line
        if self.n_labels==2:
            self.label_np[self.label_np > 0] = 1

        self.label = torch.from_numpy(np.expand_dims(self.label_np,axis=0)).long()

        # 预测结果保存
       # "Save the prediction results."
        self.result = None

    def __getitem__(self, index):
        data = torch.from_numpy(self.data_np[index])
        data = torch.FloatTensor(data).unsqueeze(0)
        return data

    def __len__(self):
        return len(self.data_np)

    def update_result(self, tensor):
        # is the shape of tensor like below
        # tensor = tensor.detach().cpu() # shape: [N,class,s,h,w]
        # tensor_np = np.squeeze(tensor_np,axis=0)
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
        else:
            self.result = tensor

    def recompone_result(self):
        # what is purpose of this function
        # shape of result: [N,class,s,h,w]
        patch_s = self.result.shape[2]
        # patch_s=


        N_patches_img = (self.padding_shape[0] - patch_s) // self.cut_stride + 1
        assert (self.result.shape[0] == N_patches_img)
        # hah, this imply that the first dim is number of patches, which is 3 in this config
        

        full_prob = torch.zeros((self.n_labels, self.padding_shape[0], self.ori_shape[1],self.ori_shape[2]))  # itialize to zero mega array with sum of Probabilities
        # (class,96,512,512)
        full_sum = torch.zeros((self.n_labels, self.padding_shape[0], self.ori_shape[1], self.ori_shape[2]))

        for s in range(N_patches_img):
            full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s] += self.result[s]
            full_sum[:, s * self.cut_stride:s * self.cut_stride + patch_s] += 1

        assert (torch.min(full_sum) >= 1.0)  # at least one
        final_avg = full_prob / full_sum
        # print(final_avg.size())
        assert (torch.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
        assert (torch.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
        # remove the padding
        img = final_avg[:, :self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]]
        return img.unsqueeze(0)

    def padding_img(self, img, size, stride):
        # default : size=48, stride=24
        assert (len(img.shape) == 3)  # 3D array
        img_s, img_h, img_w = img.shape
        leftover_s = (img_s - size) % stride
        print("left",leftover_s) # 3

        if (leftover_s != 0):
            #s= 75+24-3
            s = img_s + (stride - leftover_s)
        else:
            s = img_s

        tmp_full_imgs = np.zeros((s, img_h, img_w),dtype=np.float32)
        tmp_full_imgs[:img_s] = img
        print("Padded images shape: " + str(tmp_full_imgs.shape))
        return tmp_full_imgs
    
    # Divide all the full_imgs in pacthes
    def extract_ordered_overlap(self, img, size, stride):
        img_s, img_h, img_w = img.shape
        assert (img_s - size) % stride == 0
        N_patches_img = (img_s - size) // stride + 1
        #  N_patches_img=3, (96-48)//24 +1

        print("Patches number of the image:{}".format(N_patches_img))
        patches = np.empty((N_patches_img, size, img_h, img_w), dtype=np.float32)

        for s in range(N_patches_img):  # loop over the full images
            # cai nay hay
            patch = img[s * stride : s * stride + size]
            patches[s] = patch

        return patches  # array with all the full_imgs divided in patches

def Test_Datasets(dataset_path, args):
    data_list = sorted(glob(os.path.join(dataset_path, 'ct/*')))
    label_list = sorted(glob(os.path.join(dataset_path, 'label/*')))
    print("The number of test samples is: ", len(data_list))
    for datapath, labelpath in zip(data_list, label_list):
        print("\nStart Evaluate: ", datapath)
        # this yield look cool
        yield Img_DataSet(datapath, labelpath,args=args), datapath.split('-')[-1]

if __name__ == '__main__':
    f_ct="/root/repo/liver-tumor-segmentation/data/volume-0.nii"
    f_seg="/root/repo/liver-tumor-segmentation/data/segmentation-0.nii"
    Img_DataSet(f_ct, f_seg,args=config.args )