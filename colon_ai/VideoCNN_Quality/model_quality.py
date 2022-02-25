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
import wandb
import torchvision
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.metrics.functional import accuracy
from torchmetrics.functional import f1

from colon_ai.VideoCNN_Quality.Datamodule_Quality import VideoCNNDataModuleQuality


class VideoClassificationLightningModuleQuality(pytorch_lightning.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = self.make_kinetics_resnet()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()


    def make_kinetics_resnet(self):
        return pytorchvideo.models.resnet.create_resnet(
            input_channel=3,
            model_depth=50,
            model_num_class=3,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        f1_out = f1(y_hat, batch["label"], average='weighted', num_classes=3)
        self.log('F1_train', f1_out, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.val_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        f1_out = f1(y_hat, batch["label"], average='weighted', num_classes=3)
        self.log('F1_val', f1_out, on_epoch=True, prog_bar=True, sync_dist=True)
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
            print("shape video: ", shape(video))
            video = torch.permute(video, (0, 2, 1, 3, 4)) # b, t, c, h, w
            grid = torchvision.utils.make_grid(video[0], normalize=True)
            pl_module.logger.experiment.log({f"{prefix}_dataset": wandb.Image(grid)})


def train_part():
    seed_everything(123)

    hparams = {'weight_decay':  1.4654721473917343e-05,
               'batch_size': 3,
               'learning_rate':  0.0005723588537530765,
               'num_workers': 3,
               'gpus': 1,
               'test':1
               }
    classification_module = VideoClassificationLightningModuleQuality(hparams)
    data_module = VideoCNNDataModuleQuality(hparams)
    checkpoint_callback = ModelCheckpoint(filename='test--{epoch}-{val_loss:.2f}-{val_acc:.2f}', monitor="val_loss",
                                           verbose=True)
    trainer = pytorch_lightning.Trainer(max_epochs=30, gpus=hparams['gpus'], logger=WandbLogger(),callbacks=[Datasetview(), checkpoint_callback],log_every_n_steps=5)
    trainer.fit(classification_module, data_module)
    trainer.test(datamodule=data_module)

if __name__ == '__main__':
    print("...........Training Starts............", "\n")
    train_part()
