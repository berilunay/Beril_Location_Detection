import os
import pathlib
import matplotlib.pyplot as plt
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data

from numpy import shape
import numpy as np
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from torch.utils.data import DataLoader

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torch.utils.data import random_split
from torchvision.datasets.folder import make_dataset

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip, Resize, ColorJitter, ToTensor
)

from colon_ai.VideoCNN_Quality.labeledvideodataset import LabeledVideoDataset


class VideoCNNDataModuleQuality(pytorch_lightning.LightningDataModule):
    @property
    def hparams(self):
        return self._hparams

    def __init__(self, hparams):
        super(VideoCNNDataModuleQuality, self).__init__()
        self.hparams = hparams
        self._DATA_PATH_TRAIN = "/home/beril/Thesis_Beril/Dataset_preprocess_new/Video_Quality_Labels/quality_train"
        self._DATA_PATH_TEST = "/home/beril/Thesis_Beril/Dataset_preprocess_new/Video_Quality_Labels/quality_test"
        self._DATA_PATH_VAL = "/home/beril/Thesis_Beril/Dataset_preprocess_new/Video_Quality_Labels/quality_validation"
        #self._CLIP_DURATION = self.hparams.clip_duration
        self._CLIP_DURATION = 0.6 # Duration of sampled clip for each video 0.2 before
        self.train_transforms = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Resize(224),
                            Lambda(lambda x: x / 255.0),
                            RandomCrop(224),
                            RandomHorizontalFlip(p=0.5),


                        ]
                    ),
                ),
            ]
        )

    def dataset_path(self,dataset_path):
        dir_path = pathlib.Path(dataset_path)
        labeled_video_paths = []
        labels = ["G","B","M","p"]
        for label in labels:
            label_path = dir_path / label
            video_paths = sorted(label_path.iterdir())
            label_number = labels.index(label)

            for video_path in video_paths:
                labeled_video_paths.append((video_path, label_number))
        return labeled_video_paths

    def setup(self, stage=None):

        #labeled_video_paths = [("...Vıdeo.mpa", 0)]

        #define data_paths
        train_dataset_path = self.dataset_path(self._DATA_PATH_TRAIN)
        test_dataset_path  = self.dataset_path(self._DATA_PATH_TEST)
        val_dataset_path   = self.dataset_path(self._DATA_PATH_VAL)

        train_dataset = LabeledVideoDataset(
            labeled_video_paths=train_dataset_path,
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
            decode_audio=False,
            transform=self.train_transforms
        )

        #shape tensor 3, 15, 224, 224
        batch = next(iter(train_dataset))
        print(shape(batch["video"]))


        val_dataset=LabeledVideoDataset(
            labeled_video_paths=val_dataset_path,
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
            decode_audio=False,
            transform = self.train_transforms
        )
        test_dataset=LabeledVideoDataset(
            labeled_video_paths=test_dataset_path,
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            decode_audio=False,
            transform = self.train_transforms
        )

        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        if stage == "test" or stage is None:
            self.test_dataset = test_dataset


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'])
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'])

    @hparams.setter
    def hparams(self, value):
        self._hparams = value
