import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from glob import glob
from os.path import join

PATH_TO_DATA = "data/ADEChallengeData2016"

class ADE20KDataset(Dataset):
    def __init__(self, split="training", data_dir=PATH_TO_DATA, transform=True):
        assert (split in ["training", "validation"])
        self.dir = os.path.join(data_dir, split)
        self.transform = transform
        self.img_path = join(join(PATH_TO_DATA, "images"), split)
        self.labels_path = join(join(PATH_TO_DATA, "annotations"), split)
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
                gt = TF.hflip(label)

        img = transforms.ToTensor()(img)
        label = torch.from_numpy(np.array(label)).long()

        return img, label