import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
import sys
sys.path.append("/root/repo/help_repo/Chexpert/")
from data.imgaug import GetTransforms
from data.utils import transform
np.random.seed(0)


class ImageDataset(Dataset):
    # label_path is csv_path ?
    def __init__(self, label_path, cfg, mode='train',root_path="/root/data/"):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.root=root_path
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        with open(label_path) as f:
            #f.readline() will return the first line, which is the header
            #cool "Path,Sex,Age,Frontal/Lateral,AP/PA,No_Finding,Enlarged_Cardiomediastinum,Cardiomegaly,Lung_Opacity,Lung_Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural_Effusion,Pleural_Other,Fracture,Support_Devices"
            header = f.readline().strip('\n').split(',')

            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = os.path.join(self.root,fields[0])
                flg_enhance = False
                for index, value in enumerate(fields[5:]):
                    # seem like there is some partition about which class will update value -1 to 0, and -1 to 1
                    if index == 5 or index == 8:
                        ## get  is similar to use dict ["value"]
                        labels.append(self.dict[1].get(value))
                        # what is 
                        #"enhance_index": [2,6],  which mean it will count from class 3rd and 7th
                        if self.dict[1].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                        if self.dict[0].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                #After this code is executed, the labels list will contain the same values, but they will be converted from strings to integers. For example, if labels was ['1', '2', '3'], it will become [1, 2, 3].

                    # This type of conversion is often useful when you have data stored as strings, but you need to perform numerical operations or comparisons with the data.
                # why not use value as int in the beginning

                labels = list(map(int, labels))
                self._image_paths.append(image_path)
                # /nas/public/CheXpert/CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg
                assert os.path.exists(image_path), image_path
                self._labels.append(labels)
                if flg_enhance and self._mode == 'train':
                    for i in range(self.cfg.enhance_times):
                        self._image_paths.append(image_path)
                        self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image = Image.fromarray(image)
        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        image = np.array(image)
        image = transform(image, self.cfg)
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))

if __name__=="__main__":
    import pandas as pd
    path="/root/data/CheXpert-v1.0-small/train.csv"
    df=pd.read_csv(path)
    print (df.iloc[0,0])
    # print(df.head())
    with open(path) as f:
        # f.readline() will return the list of columns
        print (f.readline())
        header = f.readline().strip('\n').split(',')

    from easydict import EasyDict as edict
    import json
    cfg_path="/root/repo/help_repo/Chexpert/config/example_AVG.json"
    with open(cfg_path) as f:
        cfg = edict(json.load(f))
    label_path="/root/data/CheXpert-v1.0-small/train.csv"    
   
    d=ImageDataset(label_path,cfg)
   
