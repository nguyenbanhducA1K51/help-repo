from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
import numpy as np
from collections import OrderedDict
import torch
from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding_batched, resize_segmentation
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from torch.nn.functional import avg_pool2d, avg_pool3d
import numpy as np
import numpy as np


class RemoveKeyTransform(AbstractTransform):
    def __init__(self, key_to_remove):
        self.key_to_remove = key_to_remove

    def __call__(self, **data_dict):
        _ = data_dict.pop(self.key_to_remove, None)
        return data_dict


class MaskTransform(AbstractTransform):
    def __init__(self, dct_for_where_it_was_used, mask_idx_in_seg=1, set_outside_to=0, data_key="data", seg_key="seg"):
        """
        data[mask < 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.dct_for_where_it_was_used = dct_for_where_it_was_used
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning("mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist")
        data = data_dict.get(self.data_key)
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            for c in range(data.shape[1]):
                if self.dct_for_where_it_was_used[c]:
                    data[b, c][mask < 0] = self.set_outside_to
        data_dict[self.data_key] = data
        return data_dict
    
class DownsampleSegForDSTransform2(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self, ds_scales=(1, 0.5, 0.25), order=0, cval=0, input_key="seg", output_key="seg", axes=None):
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.cval = cval
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        data_dict[self.output_key] = downsample_seg_for_ds_transform2(data_dict[self.input_key], self.ds_scales, self.order, self.cval, self.axes)
        return data_dict


def downsample_seg_for_ds_transform2(seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), order=0, cval=0, axes=None):
    if axes is None:
        axes = list(range(2, len(seg.shape)))
    output = []
    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(seg)
        else:
            new_shape = np.array(seg.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            new_shape = np.round(new_shape).astype(int)
            out_seg = np.zeros(new_shape, dtype=seg.dtype)
            for b in range(seg.shape[0]):
                for c in range(seg.shape[1]):
                    out_seg[b, c] = resize_segmentation(seg[b, c], new_shape[2:], order)
            output.append(out_seg)
    return output

class DataLoader(SlimDataLoaderBase):
    def __init__(self, df, batch_size=2, patch_size = np.array([205, 205, 205]), final_patch_size = np.array([128, 128, 128]), num_batches=None, seed=False):
        super(DataLoader, self).__init__(df, batch_size, None) 
        
        # data is now stored in self._data.
        self.df = df
        
        pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = "constant"
        self.oversample_foreground_percent = 0.6
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.memmap_mode = "r"
        self.num_channels = None
        self.pad_sides = None
        
        self.need_to_pad = (np.array(self.patch_size) - np.array(self.final_patch_size)).astype(int)
        self.data_shape = (self.batch_size, 1, *self.patch_size)
        self.seg_shape = (self.batch_size, 1, *self.patch_size)
        
    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def generate_train_batch(self):
        # usually you would now select random instances of your data. We only have one therefore we skip this
        try:
            df_choice = self.df.sample(self.batch_size)
            data = np.zeros(self.data_shape, dtype=np.float32)
            seg = np.zeros(self.seg_shape, dtype=np.float32)
            i = 0
            for index, row in df_choice.iterrows():
    #             force_fg = True
                if self.get_do_oversample(i):
                    force_fg = True
                else:
                    force_fg = False

                volume = np.load(row["volume_npy"].replace('/u01/measure_ws/data_rectal/v5/preprocessed_20230508_5.0/','/raid/huyenhc/data_rectal/processed/preprocessed_20230508_5.0/'), self.memmap_mode)
                segment = np.load(row["segment_npy"].replace('/u01/measure_ws/data_rectal/v5/preprocessed_20230508_5.0/','/raid/huyenhc/data_rectal/processed/preprocessed_20230508_5.0/'), self.memmap_mode)
#                 

                need_to_pad = self.need_to_pad
                for d in range(3):
                    if need_to_pad[d] + volume.shape[d] < self.patch_size[d]:
                        need_to_pad[d] = self.patch_size[d] - volume.shape[d]

                # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
                # define what the upper and lower bound can be to then sample form them with np.random.randint
                shape = volume.shape
                lb_x = - need_to_pad[0] // 2
                ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
                lb_y = - need_to_pad[1] // 2
                ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
                lb_z = - need_to_pad[2] // 2
                ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

                if not force_fg:
#                     lb_x = np.where(np.any(volume>=-17, (1, 2)))[0][0]
#                     if lb_x>= ub_x + 1:
#                         print(volume.shape, lb_x, ub_x)
#                         ub_x = lb_x
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
                else:
                    selected_class = np.random.choice([1, 2, 3, 4])
                    if selected_class == 1:
                        voxels_of_that_class = row["location_M"]
                    elif selected_class == 2:
                        voxels_of_that_class = row["location_R"]
                    elif selected_class == 3:
                        if len(row["location_T"]) != 0:
                            voxels_of_that_class = row["location_T"]
                        else:
                            selected_class = 1
                            voxels_of_that_class = row["location_M"]
                    elif selected_class == 4:
                        if len(row["location_N"]) != 0:
                            voxels_of_that_class = row["location_N"]
                        else:
                            selected_class = 1
                            voxels_of_that_class = row["location_M"]

                    selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)

                bbox_x_ub = bbox_x_lb + self.patch_size[0]
                bbox_y_ub = bbox_y_lb + self.patch_size[1]
                bbox_z_ub = bbox_z_lb + self.patch_size[2]


                valid_bbox_x_lb = max(0, bbox_x_lb)
                valid_bbox_x_ub = min(shape[0], bbox_x_ub)
                valid_bbox_y_lb = max(0, bbox_y_lb)
                valid_bbox_y_ub = min(shape[1], bbox_y_ub)
                valid_bbox_z_lb = max(0, bbox_z_lb)
                valid_bbox_z_ub = min(shape[2], bbox_z_ub)

                volume = np.copy(volume[valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub,
                            valid_bbox_z_lb:valid_bbox_z_ub])
                segment = np.copy(segment[valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub,
                            valid_bbox_z_lb:valid_bbox_z_ub])

                

                data[i, 0] = np.pad(volume, ((-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                 self.pad_mode, **self.pad_kwargs_data)
                seg[i, 0] = np.pad(segment, ((-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                 self.pad_mode, **self.pad_kwargs_data)

                i+=1
            keys = df_choice["case"].tolist()
            del df_choice
#             np.save("data.npy", data)
#             np.save("seg.npy", seg)
#             np.save("keys.npy", np.array(keys))
        except Exception as e:
            print(e)
        return {'data': data, 'seg': seg, 'keys': keys}

from batchgenerators.transforms.channel_selection_transforms import SegChannelSelectionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.resample_transforms  import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms  import MirrorTransform
# from batchgenerators.transforms import DownsampleSegForDSTransform2, 
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, NumpyToTensor, RenameTransform
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
# from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

def get_generator(dataloader, test_dataloader, params):
    order_data = 3
    order_seg = 1
    border_val_seg = -1
    tr_transforms = []
    patch_size = params["patch_size"]
    
    if  "net_num_pool_op_kernel_sizes" in params.keys():
        net_num_pool_op_kernel_sizes = params["net_num_pool_op_kernel_sizes"]
    else:
        net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
                np.vstack(net_num_pool_op_kernel_sizes), axis=0))[:-1]

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None,
        do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
        sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
        do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=None))
    tr_transforms.append(GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=0.1))  # inverted gamma
    GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"])
    tr_transforms.append(MirrorTransform(params.get("mirror_axes")))
    tr_transforms.append(MaskTransform(params.get("use_mask_for_norm"), mask_idx_in_seg=0, set_outside_to=0))
    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(RenameTransform('seg', 'target', True))
    tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='target',
                                                                  output_key='target'))
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)
    batchgenerator_train = MultiThreadedAugmenter(dataloader, tr_transforms, params.get('num_threads'),
                                                  params.get("num_cached_per_thread"),
                                                  seeds=None, pin_memory=True)
#     batchgenerator_train = SingleThreadedAugmenter(dataloader, tr_transforms)
    
    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))
    val_transforms.append(RenameTransform('seg', 'target', True))
    val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='target',
                                                                   output_key='target'))
    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)
    
    batchgenerator_val = MultiThreadedAugmenter(test_dataloader, val_transforms,
                                                max(params.get('num_threads') // 2, 1),
                                                params.get("num_cached_per_thread"),
                                                seeds=None, pin_memory=True)
#     batchgenerator_val = SingleThreadedAugmenter(test_dataloader, val_transforms)
    
    return batchgenerator_train, batchgenerator_val

def get_class_location(row):
    row["location_M"] = np.load(row["location_M"].replace('/u01/measure_ws/data_rectal/v5/preprocessed_20230508_5.0/','/raid/huyenhc/data_rectal/processed/preprocessed_20230508_5.0/'), "r")
    row["location_R"] = np.load(row["location_R"].replace('/u01/measure_ws/data_rectal/v5/preprocessed_20230508_5.0/','/raid/huyenhc/data_rectal/processed/preprocessed_20230508_5.0/'), "r")
    row["location_T"] = np.load(row["location_T"].replace('/u01/measure_ws/data_rectal/v5/preprocessed_20230508_5.0/','/raid/huyenhc/data_rectal/processed/preprocessed_20230508_5.0/'), "r")
    row["location_N"] = np.load(row["location_N"].replace('/u01/measure_ws/data_rectal/v5/preprocessed_20230508_5.0/','/raid/huyenhc/data_rectal/processed/preprocessed_20230508_5.0/'), "r")
    return row


import pandas as pd
import os
from sklearn.model_selection import train_test_split
class CustomNnunetData:
    def __init__(self, configs):
        input_folder = configs.dataset["input_folder"]
        output_folder = configs.output_folder
        params = configs.dataset["params"]

        self.params = params
#         train_df = pd.read_csv(os.path.join(input_folder, "data_train_public_manual.csv"))
#         val_df = pd.read_csv(os.path.join(input_folder, "data_test_rm_outliers.csv"))
        
        train_path = os.path.join(output_folder, "train.csv")
        val_path = os.path.join(output_folder, "test.csv")
        
        if not os.path.exists(train_path):
            os.makedirs(output_folder, exist_ok=True)
            df = pd.read_csv(os.path.join(input_folder, "train.csv"))
            train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
        else:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            train_df = train_df[~train_df['vol_path'].isna()]
            val_df = val_df[~val_df['vol_path'].isna()]
            print('train_df.shape=',train_df.shape)
            print('val_df.shape=',val_df.shape)
        test_df = pd.read_csv(os.path.join(input_folder, "test.csv"))
        test_df = test_df[~test_df['vol_path'].isna()]
        print('test_df.shape=',test_df.shape)
        self.train_df = train_df.apply(get_class_location, axis = 1)
        self.val_df = val_df.apply(get_class_location, axis = 1)
        self.test_df = test_df.apply(get_class_location, axis = 1)
        
        dataloader = DataLoader(self.train_df, batch_size=configs.batch_size, patch_size=params['patch_size_for_spatialtransform'])
        val_dataloader = DataLoader(self.val_df, patch_size=params["patch_size"], batch_size= 2)
        test_dataloader = DataLoader(self.test_df, patch_size=params["patch_size"], batch_size= 2)
        
#         self.dataloader = dataloader
#         self.val_dataloader = val_dataloader
        
        self.gen_train, self.gen_val = get_generator(dataloader, val_dataloader, params)
        self.gen_train, self.gen_test = get_generator(dataloader, test_dataloader, params)
        
        self.train_loader = np.arange(configs.dataset["steps_per_epoch"])
        self.val_loader = np.arange(25)
        self.test_loader = np.arange(25)
                
        print('----Input folder----', input_folder)
        print('----Output folder----', output_folder)
        print('No. thread:', params.get('num_threads'), '- No. cache per thread:',params.get("num_cached_per_thread"))
        print('No. training data:', len(self.train_df))
if __name__=="__main__":
    
    import configs.config_v5_v3 as configs
    data = CustomNnunetData(configs)
        

        