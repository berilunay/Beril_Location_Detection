import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
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

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

class VideoCNNDataModule(pytorch_lightning.LightningDataModule):
   def __init__(self, hparams):
     super(VideoCNNDataModule, self).__init__()
     self.hparams= hparams
     self._DATA_PATH ='/home/beril/Thesis_Beril/Dataset_VideoCNN/Train_Location_Video'
     self._CLIP_DURATION = 2  # Duration of sampled clip for each video
     self.train_transforms=Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(8),
                        Lambda(lambda x: x / 255.0),
                        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(244),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

   def setup(self, stage=None):

       train_dataset = pytorchvideo.data.LabeledVideoDataset(
        labeled_video_paths=self._DATA_PATH,
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
        transform=self.train_transforms)

       # do the split
       len_train_dataset = len(train_dataset)
       len_train_splitted = int(0.6 * len_train_dataset)
       len_val = len_train_dataset - len_train_splitted

       train_dataset, val_dataset = random_split(train_dataset, [len_train_splitted, len_val])

       # do the splıt agaın to splıt val ın val + test
       len_val_dataset = len(val_dataset)
       len_val_splitted = int(0.5 * len_val_dataset)
       len_test = len_val_dataset - len_val_splitted

       val_dataset, test_dataset = random_split(val_dataset, [len_val_splitted, len_test])

       if stage == "fit" or stage is None:
           self.train_dataset = train_dataset
           self.val_dataset = val_dataset

       if stage == "test" or stage is None:
           self.test_dataset = test_dataset


   def train_dataloader(self):
       return DataLoader(self.train_dataset,batch_size=self.hparams.batch_size,num_workers=self.hparams.num_workers)

   def val_dataloader(self):
       return DataLoader(self.val_dataset,batch_size=self.hparams.batch_size,num_workers=self.hparams.num_workers)

   def test_dataloader(self):
       return DataLoader(self.test_dataset,batch_size=self.hparams.batch_size,num_workers=self.hparams.num_workers)