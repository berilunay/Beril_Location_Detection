import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from DatasetClass import ColonDataset


class ColonDataModule(pl.LightningDataModule):
    @property
    def hparams(self):
        return self._hparams

    def __init__(self, hparams, mean=None, std=None):
        super(ColonDataModule, self).__init__()
        #self.save_hyperparameters(hparams)
        self.hparams = hparams
        self.root_dir_train = "/home/beril/Thesis_Beril/Train_Labels_Quality"

        self.my_transform = transforms.Compose([
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=95),
            transforms.ColorJitter(brightness=0.5)
        ])

    def setup(self, stage=None):

        train_dataset = ColonDataset(root=self.root_dir_train, transform=self.my_transform)

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
        #return DataLoader(self.train_dataset, self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return DataLoader(self.train_dataset, self.hparams["batch_size"], num_workers=self.hparams["num_workers"])


    def val_dataloader(self):
        #return DataLoader(self.val_dataset, self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return DataLoader(self.val_dataset,  self.hparams["batch_size"],num_workers=self.hparams["num_workers"])
    def test_dataloader(self):
        #return DataLoader(self.test_dataset, self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return DataLoader(self.test_dataset,  self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    @hparams.setter
    def hparams(self, value):
        self._hparams = value
