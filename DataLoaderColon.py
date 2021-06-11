import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from DatasetClass import ColonDataset




class ColonDataModule(pl.LightningDataModule):
    def __init__(self,hparams, mean=None, std=None):
        super(ColonDataModule, self).__init__()
        self.hparams = hparams

        self.root_dir_train= "D:\Beril\Thesis\Data\Train_Label\Labels"
        self.root_dir_test = "D:\Beril\Thesis\Data\Train_Label\Labels"  #create new folder in the server
        self.transform = transforms.Normalize(mean, std) if mean is not None else None


    def setup(self, stage=None):

        train_dataset=ColonDataset(root=self.root_dir_train,transform=self.transform)
        test_dataset =ColonDataset(root=self.root_dir_test,transform=self.transform)

        #do the split
        len_train_dataset = len(train_dataset)
        len_train_splitted = int(0.75 * len_train_dataset)
        len_val = len_train_dataset - len_train_splitted

        train_dataset, val_dataset = random_split(train_dataset, [len_train_splitted, len_val])

        if stage == "fit" or stage is None:
                self.train_dataset = train_dataset
                self.val_dataset = val_dataset

        if stage == "test" or stage is None:
                self.test_dataset = test_dataset


    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.hparams.batch_size)