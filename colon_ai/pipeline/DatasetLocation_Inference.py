from torchvision.datasets import VisionDataset
import torch
import numpy as np
from pathlib import Path
import cv2

class InferenceDatasetLocation(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, num_samples=None):
        super(InferenceDatasetLocation, self).__init__(root, transform=transform, target_transform=target_transform)
        self.video_dirs = []
        self.video_dirs.append(self.root)
        self.sample_dirs = []  # image0,image1 ...
        for video_dir in self.video_dirs:
            self.sample_dirs += (sorted(Path(video_dir).iterdir())) # gets image and labels(folder)
        if num_samples:
            self.sample_dirs = self.sample_dirs[:num_samples]



    def __getitem__(self, index):
        sample_dir = self.sample_dirs[index]
        colon= self._load_and_transform_colon(sample_dir)
        return colon


    def _load_and_transform_colon(self, sample_dir):
        colon = self._load_colon(sample_dir)
        colon = self._apply_transforms(colon)

        return colon


    def _load_colon(self, sample_dir):

        im_path = str(sample_dir / "3D.png")
        colon=cv2.imread(im_path)
        resized_image = cv2.resize(colon, (224, 224),interpolation=cv2.INTER_NEAREST)
        img_new=cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        colon = np.array(img_new)
        colon=np.moveaxis(colon,-1,0)
        colon = torch.from_numpy(colon)
        colon= colon/255.0

        return colon


    def _apply_transforms(self, colon):
        if self.transform is not None:
            colon = self.transform(colon)

        return colon


    def __len__(self):
        return len(self.sample_dirs)