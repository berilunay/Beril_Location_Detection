import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

from colon_ai.train_location.DatasetClass_Location import ColonDatasetLocation


class ColonDataModuleLocation(pl.LightningDataModule):
    @property
    def hparams(self):
        return self._hparams

    def __init__(self, hparams, mean=None, std=None):
        super(ColonDataModuleLocation, self).__init__()
        #self.save_hyperparameters(hparams)
        self.hparams = hparams
        self.root_dir_train = "/home/beril/Thesis_Beril/Dataset_preprocess_new/Location_Detection/train_location_labels"
        self.root_dir_test="/home/beril/Thesis_Beril/Dataset_preprocess_new/Location_Detection/test_location_labels"
        self.root_dir_val = "/home/beril/Thesis_Beril/Dataset_preprocess_new/Location_Detection/val_location_labels"
        #train w/ less data
        # self.root_dir_train = "/home/beril/Thesis_Beril/Train_Labels_location"
        # self.root_dir_test="/home/beril/Thesis_Beril/val_labels_location"
        # self.root_dir_val = "/home/beril/Thesis_Beril/test_labels_location"

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

        train_dataset = ColonDatasetLocation(root=self.root_dir_train,transform=self.val_test_transform)
        val_dataset = ColonDatasetLocation(root=self.root_dir_val,transform=self.val_test_transform)
        test_dataset = ColonDatasetLocation(root=self.root_dir_test,transform=self.val_test_transform)

        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        if stage == "test" or stage is None:
            self.test_dataset = test_dataset

        print('Train data set:', len(train_dataset))
        print('Test data set:', len(test_dataset))
        print('Valid data set:', len(val_dataset))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.hparams["batch_size"], shuffle=True,num_workers=self.hparams["num_workers"])


    def val_dataloader(self):
        return DataLoader(self.val_dataset,  self.hparams["batch_size"],num_workers=self.hparams["num_workers"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset,  self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    @hparams.setter
    def hparams(self, value):
        self._hparams = value
