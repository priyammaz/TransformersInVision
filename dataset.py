import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from glob import glob
import os
from os.path import join
from PIL import Image
import numpy as np

PATH_TO_DATA_ADE20K = "data/ADEChallengeData2016"
PATH_TO_DATA_MP4 = "data/mp4"

class ADE20KDataset(Dataset):
    def __init__(self, split="training", data_dir=PATH_TO_DATA_ADE20K, transform=True):
        assert (split in ["training", "validation"])
        self.dir = join(data_dir, split)
        self.transform = transform
        self.img_path = join(join(data_dir, "images"), split)
        self.labels_path = join(join(data_dir, "annotations"), split)
        self.images = glob(join(self.img_path, "*.jpg"), recursive=True)
        self.labels = [self.get_image_label(img) for img in self.images]
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def get_image_label(self, img_path):
        img_path = img_path.split("/")[-1][:-3] + "png"
        return join(self.labels_path, img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])

        if self.transform:
            resize = transforms.Resize(size=(512, 512))
            img = resize(img)
            img = self.normalize(img)
            label = resize(label)
            l, r, h, w = transforms.RandomCrop.get_params(img, output_size=(256, 256))
            img = TF.crop(img, l, r, h, w)
            label = TF.crop(label, l, r, h, w)

            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
                label = TF.hflip(label)

        img = transforms.ToTensor()(img)
        label = torch.from_numpy(np.array(label)).long()

        return img, label


class SegmentationDataset(Dataset):
    """
    Data loader for the Segmentation Dataset. If data loading is a bottleneck,
    you may want to optimize this in for faster training. Possibilities include
    pre-loading all images and annotations into memory before training, so as
    to limit delays due to disk reads.
    """

    def __init__(self, split="train", data_dir=PATH_TO_DATA_MP4, transform=False):
        assert (split in ["train", "val", "test"])
        self.img_dir = os.path.join(data_dir, split)
        self.classes = []
        with open(os.path.join(data_dir, 'classes.txt'), 'r') as f:
            for l in f:
                self.classes.append(l.rstrip())
        self.n_classes = len(self.classes)
        self.split = split
        self.data = glob.glob(self.img_dir + '/*.jpg')
        self.data = [os.path.splitext(l)[0] for l in self.data]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index] + '.jpg')
        gt = Image.open(self.data[index] + '.png')

        if self.transform:
            ow, oh = img.size
            l, r, h, w = transforms.RandomCrop.get_params(img, output_size=(128, 128))
            img = transforms.Resize(size=(oh, ow))(TF.crop(img, l, r, h, w))
            gt = transforms.Resize(size=(oh, ow))(TF.crop(gt, l, r, h, w))

            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
                gt = TF.hflip(gt)

            blurrer = transforms.GaussianBlur(kernel_size=(5, 5))
            img = blurrer(img)
        img = transforms.ToTensor()(img)
        gt = torch.squeeze(transforms.PILToTensor()(gt)).type(torch.LongTensor)
        return img, gt