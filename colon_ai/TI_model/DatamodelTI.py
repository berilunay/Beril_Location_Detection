import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, wandb
from pytorch_lightning.metrics.functional import accuracy
from torchvision.models import resnet18
from pytorch_lightning import LightningModule, Callback
from pytorch_lightning.metrics import MeanAbsoluteError
from torch.nn.functional import l1_loss
from torch.optim import Adam
from torch.nn import functional as F
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from torchmetrics import F1

from argparse import ArgumentParser
import wandb

from colon_ai.TI_model.DataLoaderTI import ColonDataModuleTI
from colon_ai.traınıng.DataLoaderColon import ColonDataModule


class ColonDataModel_TI(LightningModule):
    def __init__(self, hparams):
        super(ColonDataModel_TI, self).__init__()
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
        # f1 = F1(num_classes=4)
        # f1_out=f1(preds, targets)
        # self.log('F1',f1_out)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss


    def validation_step(self, batch, batch_idx):
        images, targets = batch
        out = self.forward(images)
        loss = F.cross_entropy(out, targets)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, targets)
        # f1 = F1(num_classes=4)
        # f1_out=f1(preds, targets)
        # self.log('F1',f1_out)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss


    def test_step(self, batch, batch_idx):
        images, targets = batch
        out = self.forward(images)
        loss = F.cross_entropy(out, targets)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, targets)
        # f1 = F1(num_classes=4)
        # f1_out=f1(preds, targets)
        # self.log('F1',f1_out)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return loss


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams["learning_rate"],weight_decay=self.hparams["weight_decay"])
        return optimizer



def show_examples(model, datamodule, class_dict=None):
    data_loader=datamodule.test_dataloader()
    for batch_idx, (features, targets) in enumerate(data_loader):
        with torch.no_grad():
            features = features
            targets = targets
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)
        break

    fig, axes = plt.subplots(nrows=4, ncols=5,
                             sharex=True, sharey=True)


    nhwc_img = np.transpose(features, axes=(0, 2, 3, 1))

    if nhwc_img.shape[-1] == 1:
        nhw_img = np.squeeze(nhwc_img.numpy(), axis=3)

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhw_img[idx], cmap='binary')
            if class_dict is not None:
                ax.title.set_text(f'P: {class_dict[predictions[idx].item()]}'
                                  f'\nT: {class_dict[targets[idx].item()]}')
            else:
                ax.title.set_text(f'P: {predictions[idx]} | T: {targets[idx]}')
            ax.axison = False

    else:

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhwc_img[idx])
            if class_dict is not None:
                ax.title.set_text(f'P: {class_dict[predictions[idx].item()]}'
                                  f'\nT: {class_dict[targets[idx].item()]}')
            else:
                ax.title.set_text(f'P: {predictions[idx]} | T: {targets[idx]}')
            ax.axison = False
    plt.tight_layout()
    plt.show()


class Datasetview2D(Callback):
    """Logs one batch of each dataloader to WandB"""

    def on_train_start(self, trainer, pl_module):
        data_loaders = {
            "train": pl_module.train_dataloader(),
            "val": pl_module.val_dataloader(),
         }

        for prefix, data_loader in data_loaders.items():
            sample_batch, target_batch = next(iter(data_loader))
            print("sample batch:",np.shape(sample_batch))
            grid = torchvision.utils.make_grid(sample_batch)

            pl_module.logger.experiment.log({f"{prefix}_dataset": wandb.Image(grid)})

def train_part():
    seed_everything(123)

    location_dict = {0:'TI',1:'p',2:'N'}

    hparams = {'weight_decay': 2.810273007555912e-05,
               'batch_size': 54,
               'learning_rate': 0.00011441600362623002,
               'num_workers': 4,
               'gpus': 1,
               'test': 1
               }
    model=ColonDataModel_TI(hparams)
    datamodule_colon=ColonDataModuleTI(hparams)

    #-----------------------------------------------------------------------------------------------------------
    checkpoint_callback = ModelCheckpoint(filename='trial--{epoch}-{val_loss:.2f}-{val_acc:.2f}--{train_loss:.2f}-{train_acc:.2f}', monitor="val_loss", verbose=True)

    trainer=Trainer( max_epochs=20, gpus=hparams["gpus"], logger=WandbLogger(), callbacks=[Datasetview2D(), checkpoint_callback], log_every_n_steps=5)
    trainer.fit(model,datamodule_colon)
    trainer.test(datamodule=datamodule_colon)

    show_examples(model,datamodule_colon,class_dict=location_dict)

    #---------------------------------------------------
    # trainer = Trainer(gpus=hparams["gpus"])
    # checkpoint_model_path_loc = "/home/beril/BerilCodes/ColonAI_LocationDetection/colon_ai/traınıng/uncategorized/1kage7r2/checkpoints/trial--epoch=29-val_loss=0.08-val_acc=0.97--train_loss=0.05-train_acc=0.98.ckpt"
    # pretrained_model_qua = ColonDataModel.load_from_checkpoint(checkpoint_path=checkpoint_model_path_loc)
    # pretrained_model_qua.eval()
    # result=trainer.test(pretrained_model_qua,datamodule=datamodule_colon)
    # show_examples(model, datamodule_colon, class_dict=location_dict)


if __name__ == '__main__':
    print("...........Training Starts............", "\n")
    train_part()