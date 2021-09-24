import torch
from PIL import Image
import numpy as np
import os.path
from pathlib import Path

from matplotlib import pyplot as plt
from torchvision.datasets import VisionDataset
import cv2


class ColonDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, num_samples=None):
        super(ColonDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.video_dirs = sorted(Path(self.root).iterdir())
        self.sample_dirs = []  # image0,image1 ...
        for video_dir in self.video_dirs:
            self.sample_dirs+=(sorted(Path(video_dir).iterdir()))  # gets image and labels(folder)

        if num_samples:
            self.sample_dirs = self.sample_dirs[:num_samples]


    def __getitem__(self, index):
        sample_dir = self.sample_dirs[index]

        colon, location = self._load_and_transform_colon_and_location(sample_dir)

        return colon, location


    def _load_and_transform_colon_and_location(self, sample_dir):
        colon = self._load_colon(sample_dir)
        location = self._load_location(sample_dir)
        colon, location = self._apply_transforms(colon, location)

        return colon, location


    def _load_colon(self, sample_dir):
        # use pil to load image. path is te sampledir
        im_path = str(sample_dir / "colon.png")
        #im_path = str(sample_dir / "3D.png")
        colon=cv2.imread(im_path)
        resized_image = cv2.resize(colon, (224, 224),interpolation=cv2.INTER_NEAREST)
        img_new=cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        colon = np.array(img_new)
        colon=np.moveaxis(colon,-1,0)
        colon = torch.from_numpy(colon)
        colon= colon/255.0

        return colon


    def _load_location(self, sample_dir):
        # check
        #location_path=str(sample_dir / "Location.txt")
        location_path = str(sample_dir / "Quality.txt")
        loc = open(location_path, 'r')
        location = loc.read()
        #location_dict= {"R":0,"L":2,"M":1}
        location_dict = {"G": 0, "p": 2, "M": 1, "B":3}
        location_label=location_dict[location]
        location_label=torch.tensor(location_label)
        return location_label


    def _apply_transforms(self, colon, location):
        if self.transform is not None:
            colon = self.transform(colon)

        if self.target_transform is not None:
            location = self.target_transform(location)

        return colon, location


    def __len__(self):
        return len(self.sample_dirs)



if __name__ == "__main__":
    print("hello")
    dataset = ColonDataset('/home/beril/Thesis_Beril/Train_Labels')
    colon,location= dataset[0]



