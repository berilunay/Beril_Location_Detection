import pytorch_lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorchvideo.models.resnet
from argparse import ArgumentParser

from numpy import shape
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from torchmetrics import Accuracy
import numpy as np

import torchvision
from pytorch_lightning.callbacks import Callback
import wandb
from torchmetrics.functional import f1

from colon_ai.VideoCNN.Datamodule import VideoCNNDataModule


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = self.make_kinetics_resnet()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()


    def make_kinetics_resnet(self):
        return pytorchvideo.models.resnet.create_resnet(
            input_channel=3,  # RGB input from Kinetics
            model_depth=50,  # For the tutorial let's just use a 50 layer network
            model_num_class=3,  # Kinetics has 400 classes so we need out final head to align
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        f1_out = f1(y_hat, batch["label"], average='weighted', num_classes=3)
        self.log('F1_train', f1_out)
        self.log("train_loss", loss)
        # self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        # probs = F.softmax(y_hat, dim=-1)
        # for index in probs:
        #     print("max prob class: ",torch.argmax(index))
        acc = self.val_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        f1_out = f1(y_hat, batch["label"], average='weighted', num_classes=3)
        self.log('F1_val', f1_out)
        self.log("val_loss", loss)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.test_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        f1_out = f1(y_hat, batch["label"], average='weighted', num_classes=3)
        self.log('F1_test', f1_out)
        self.log("test_loss", loss)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """

        return torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'],
                                weight_decay=self.hparams['weight_decay'])

class Datasetview(Callback):
    """Logs one batch of each dataloader to WandB"""

    def on_train_start(self, trainer, pl_module):
        data_loaders = {
            "train": pl_module.train_dataloader(),
            "val": pl_module.val_dataloader(),
         }

        for prefix, data_loader in data_loaders.items():
            batch = next(iter(data_loader))
            video = batch["video"]
            print("shape: ", shape(video))

            video = torch.permute(video, (0, 2, 1, 3, 4)) # b, t, c, h, w

            grid = torchvision.utils.make_grid(video[0], normalize=True)

            pl_module.logger.experiment.log({f"{prefix}_dataset": wandb.Image(grid)})




def train_part():
    seed_everything(123)


    hparams = {'weight_decay': 0.00015580008840493994,
               'batch_size': 4,
               'learning_rate': 4.051572326186531e-05,
               'num_workers': 3,
               'gpus': 1,
               'test':1
               }
    classification_module = VideoClassificationLightningModule(hparams)
    data_module = VideoCNNDataModule(hparams)
    trainer = pytorch_lightning.Trainer(max_epochs=100, gpus=hparams['gpus'], logger=WandbLogger(),callbacks=Datasetview())

    trainer.fit(classification_module, data_module)
    trainer.test(datamodule=data_module)

    # ---------------------------------------------------------------------------------------------------
    # classification_module = VideoClassificationLightningModule(hparams=args)
    # data_module = VideoCNNDataModule(hparams=args)
    #
    # trainer= pytorch_lightning.Trainer(max_epochs=args.max_epochs, gpus=args.gpus, logger=WandbLogger())
    # trainer.fit(classification_module, data_module)
    # trainer.test(datamodule=data_module)


if __name__ == '__main__':
    print("...........Training Starts............", "\n")
    train_part()
