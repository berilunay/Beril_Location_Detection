import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

from colon_ai.traınıng.DatasetClass import ColonDataset


class ColonDataModule(pl.LightningDataModule):
    @property
    def hparams(self):
        return self._hparams

    def __init__(self, hparams, mean=None, std=None):
        super(ColonDataModule, self).__init__()

        self.hparams = hparams
        self.root_dir_train = "/home/beril/Thesis_Beril/Dataset_preprocess_new/Quality_Detection/train_quality_Labels"
        self.root_dir_val = "/home/beril/Thesis_Beril/Dataset_preprocess_new/Quality_Detection/val_quality_labels"
        self.root_dir_test = "/home/beril/Thesis_Beril/Dataset_preprocess_new/Quality_Detection/test_quality_labels"
        #prev model
        # self.root_dir_train = "/home/beril/Thesis_Beril/Train_Labels_Quality"
        # self.root_dir_val = "/home/beril/Thesis_Beril/val_labels_quality"
        # self.root_dir_test = "/home/beril/Thesis_Beril/test_labels_quality"

        self.my_transform = transforms.Compose([
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=95),
            transforms.ColorJitter(brightness=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


        self.val_test_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):

        # train_dataset = ColonDataset(root=self.root_dir_train,transform=self.my_transform)
        # val_dataset = ColonDataset(root=self.root_dir_val,transform=self.val_test_transform)
        # test_dataset=ColonDataset(root=self.root_dir_test,transform=self.val_test_transform)

        train_dataset = ColonDataset(root=self.root_dir_train, transform=self.val_test_transform)
        val_dataset = ColonDataset(root=self.root_dir_val, transform=self.val_test_transform)
        test_dataset = ColonDataset(root=self.root_dir_test, transform=self.val_test_transform)


        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        if stage == "test" or stage is None:
            self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.hparams["batch_size"],shuffle=True, num_workers=self.hparams["num_workers"])


    def val_dataloader(self):
        return DataLoader(self.val_dataset,  self.hparams["batch_size"],num_workers=self.hparams["num_workers"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset,  self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    @hparams.setter
    def hparams(self, value):
        self._hparams = value
