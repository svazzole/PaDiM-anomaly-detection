import os
# import tarfile
from PIL import Image
from skimage.color.colorconv import gray2rgb
from torchvision.transforms.transforms import ColorJitter
from tqdm import tqdm
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = ['alldata', 'scratch', 'dotparticle', 'misprint', ]

class CFRPDataset(Dataset):
    def __init__(self, dataset_path='cfrp', class_name='scartch', is_train=True,
                resize=256, cropsize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        # self.x, self.y, self.mask = self.load_dataset_folder()
        self.x, self.y = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                    # T.ColorJitter(
                                    #     brightness=0.2,
                                    #     contrast=0.2,
                                    #     saturation=0.2,
                                    # ),
                                    T.RandomHorizontalFlip(),
                                    T.RandomVerticalFlip(),
                                    T.CenterCrop(cropsize),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]) # imagenet mean & stddev
        self.transform_test_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                            T.CenterCrop(cropsize),
                                            T.ToTensor(),
                                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
        # self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
        #                                 T.CenterCrop(cropsize),
        #                                 T.ToTensor()])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        x = Image.open(x).convert('RGB')
        if self.is_train:
            x = self.transform_x(x)
        else:
            T1 = T.RandomVerticalFlip(p=1)
            T2 = T.RandomHorizontalFlip(p=1)
            x = self.transform_test_x(x)
            x1 = T1.forward(x)
            x2 = T2.forward(x)
            x3 = T2.forward(T1.forward(x))
            x = torch.cat([x, x1, x2, x3], dim=0)
            # y = [y, y, y, y]

        return x, y

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y = [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)
                                    if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                
        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y)