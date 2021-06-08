import os
from PIL import Image
from skimage.color.colorconv import gray2rgb
from torch.utils import data
from torchvision.transforms.transforms import ColorJitter, RandomHorizontalFlip, RandomVerticalFlip
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASS_NAMES = ['scratch', 'dotparticle', 'misprint', 'alldata'] # ['scratch', 'dotparticle', 'misprint'] # 모든 데이터셋 적용할 수 있도록하면 class_name을 리스트로 하기

class CFRPDataset(Dataset):
    def __init__(self, dataset_path='dataset', class_name='scratch', is_train=True, resize=256, cropsize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)

        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        self.x, self.y = self.load_dataset_folder()

        self.train_transform = T.Compose([
            T.Resize(resize, Image.ANTIALIAS),
            # T.ColorJitter(
            #     brightness=0.2,
            #     contrast=0.2,
            #     saturation=0.2,
            # ),
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            T.CenterCrop(cropsize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = T.Compose([
            T.Resize(resize, Image.ANTIALIAS),
            T.CenterCrop(cropsize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image, label = self.x[index], self.y[index]

        image = Image.open(image).convert('RGB')
        if self.is_train:
            image = self.train_transform(image)
        else:
            TF1 = T.RandomVerticalFlip(p=1)
            TF2 = T.RandomHorizontalFlip(p=1)
            image = self.test_transform(image)
            image2 = TF1.forward(image)
            image3 = TF2.forward(image)
            image4 = TF2.forward(TF1.forward(image))
            image = torch.cat([image, image2, image3, image4], dim=0)

        return image, label

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y = [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase) # ex) img_dir/scratch/train
        img_types = sorted(os.listdir(img_dir))

        for img_type in img_types:
            # load_images
            img_type_dir = os.path.join(img_dir, img_type)

            if not os.path.isdir(img_type_dir):
                continue

            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
            
            # extend list
            x.extend(img_fpath_list)

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
            elif img_type == 'defect':
                y.extend([1] * len(img_fpath_list))
            
            assert len(x) == len(y), 'number of x and y should be same'

        return x, y # list(x), list(y)

    def denormalize(x: torch.tensor):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean)
        std = torch.tensor(std)

        return x.mul_(std).add_(mean)

if __name__ == '__main__':
    cfrp = CFRPDataset()
    cfrp.load_dataset_folder()

    print(len(cfrp))
