import pytorch_lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorchvideo.models.resnet
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from torchmetrics import Accuracy

from Datamodule import VideoCNNDataModule


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
        self.log("train_loss", loss.item())
        #self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True,)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.val_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("val_loss", loss)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True,)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.test_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("test_loss", loss)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True,)
        return loss

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def args_part():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--test", default=1, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    args = parser.parse_args()

    return args


def train_part():
    seed_everything(123)
    args = args_part()

    classification_module = VideoClassificationLightningModule(hparams=args)
    data_module = VideoCNNDataModule(hparams=args)

    trainer = pytorch_lightning.Trainer(max_epochs=args.max_epochs, gpus=args.gpus, logger=WandbLogger())
    trainer.fit(classification_module, data_module)
    trainer.test(datamodule=data_module)


if __name__ == '__main__':
    print("...........Training Starts............", "\n")
    train_part()
