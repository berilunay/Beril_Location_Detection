import torch
from pytorch_lightning.metrics.functional import accuracy
from torchvision.models import resnet18
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import MeanAbsoluteError
from torch.nn.functional import l1_loss
from torch.optim import Adam
from torch.nn import functional as F
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from DataLoaderColon import ColonDataModule
from argparse import ArgumentParser


class ColonDataModel(LightningModule):
    def __init__(self, hparams):
        super(ColonDataModel, self).__init__()
        self.save_hyperparameters(hparams)

        # Network
        self.network = resnet18(num_classes=3)


    def forward(self, x):

        return self.network(x)


    def training_step(self, batch, batch_idx):
        images,targets=batch
        out= self.forward(images)
        loss= F.cross_entropy(out,targets)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, targets)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss


    def validation_step(self, batch, batch_idx):
        images, targets = batch
        out = self.forward(images)
        loss = F.cross_entropy(out, targets)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, targets)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss


    def test_step(self, batch, batch_idx):
        images, targets = batch
        out = self.forward(images)
        loss = F.cross_entropy(out, targets)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, targets)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return loss


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def args_part():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--test", default=1, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epochs", default=15, type=int)
    args = parser.parse_args()

    return args

def train_part():
    seed_everything(123)
    args=args_part()

    model=ColonDataModel(hparams=args)
    datamodule_colon=ColonDataModule(hparams=args)

    trainer=Trainer( max_epochs=args.max_epochs)
    trainer.fit(model,datamodule_colon)

    #for testing
    if args.test:
       trainer.test(datamodule=datamodule_colon)


if __name__ == '__main__':
    print("...........Training Starts............", "\n")
    train_part()