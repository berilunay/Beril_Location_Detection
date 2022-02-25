import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, wandb
from pytorch_lightning.metrics.functional import accuracy
from torch import nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from pytorch_lightning import LightningModule, Callback
from pytorch_lightning.metrics import MeanAbsoluteError
from torch.nn.functional import l1_loss
from torch.optim import Adam, SGD
from torch.nn import functional as F
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from torchmetrics import F1
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import wandb
import pandas as pd
import seaborn
from colon_ai.TI_model.DataLoaderTI import ColonDataModuleTI
from torchmetrics.functional import f1
import torch.multiprocessing
from colon_ai.TI_model.DatasetClassTI import ColonDatasetTI



class ColonModule_TI(LightningModule):
    def __init__(self, hparams):
        super(ColonModule_TI, self).__init__()
        self.save_hyperparameters(hparams)

        """Network"""
        self.network = resnet18(pretrained=True)
        self.num_ftr = self.network.fc.in_features
        self.network.fc = nn.Linear(self.num_ftr, 3)


    def forward(self, x):

        return self.network(x)


    def training_step(self, batch, batch_idx):
        images,targets=batch
        out= self.forward(images)
        loss= F.cross_entropy(out,targets)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, targets)
        f1_out = f1(preds, targets, average='weighted', num_classes=3)
        self.log('F1_train', f1_out)
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss


    def validation_step(self, batch, batch_idx):
        images, targets = batch
        out = self.forward(images)
        loss = F.cross_entropy(out, targets)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, targets)
        f1_out = f1(preds, targets, average='weighted', num_classes=3)
        self.log('F1_val', f1_out)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

        return loss


    def test_step(self, batch, batch_idx):
        images, targets = batch
        out = self.forward(images)
        loss = F.cross_entropy(out, targets)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, targets)
        f1_out = f1(preds, targets, average='weighted', num_classes=3)
        self.log('F1_test', f1_out)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return loss


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams["learning_rate"],weight_decay=self.hparams["weight_decay"])
        return optimizer



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


def plot_conf_matrix(model,dataloader):
    orig_labels=[]
    pred_labels=[]

    for features,targets in dataloader:
        with torch.no_grad():
            targets_np=targets.numpy()
            orig_labels.append(targets_np)
            preds = model(features)
            predictions = torch.argmax(preds, dim=1)
            pred_numpy = predictions.numpy()
            pred_labels.append(pred_numpy)

    orig_labels_conv=np.concatenate(orig_labels,axis=None)
    pred_labels_conv = np.concatenate(pred_labels, axis=None)
    conf_matrix =confusion_matrix(orig_labels_conv, pred_labels_conv)
    print("conf_matrix_test: ",conf_matrix)
    df_cm = pd.DataFrame(conf_matrix, index=[i for i in "TPN"],
                         columns=[i for i in "TPN"])
    plt.figure(figsize=(3, 3))
    seaborn.heatmap(df_cm, annot=True,cmap="OrRd")
    plt.savefig("conf_matrix.png")
    plt.show()


def train_part():
    seed_everything(123)

    hparams = {'weight_decay':6.935071336760909e-05,
               'batch_size':  23,
               'learning_rate':  0.0008955313299796573,
               'num_workers': 2,
               'gpus': 1,
               'test': 1
               }

    TI_module=ColonModule_TI(hparams)
    datamodule_colon=ColonDataModuleTI(hparams)

    checkpoint_callback = ModelCheckpoint(filename='testadam--{epoch}-{val_loss:.2f}-{val_acc:.2f}--{train_loss:.2f}-{train_acc:.2f}--{F1_val:.2f}-{F1_train:.2f}', monitor="val_loss", verbose=True)
    trainer=Trainer( max_epochs=10, gpus=hparams["gpus"], logger=WandbLogger(), callbacks=[Datasetview2D(), checkpoint_callback], log_every_n_steps=5)
    trainer.fit(TI_module,datamodule_colon)

    """The part for getting the test results and the confusion matrix via inference. Should be commented out after the training is done"""
    Test_Path = "/home/beril/Thesis_Beril/Dataset_preprocess_new/procedure_detection/Test_TI_Labels"
    val_test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    checkpoint_model_path_loc = "BEST_MODEL_PATH"
    pretrained_model_TI = ColonModule_TI.load_from_checkpoint(checkpoint_path=checkpoint_model_path_loc)
    pretrained_model_TI.eval()
    trainer = Trainer(gpus=hparams["gpus"])
    trainer.test(pretrained_model_TI,datamodule=datamodule_colon)
    TI_dataset = ColonDatasetTI(root=Test_Path, transform=val_test_transform)
    dataloader_TI = DataLoader(TI_dataset, batch_size=pretrained_model_TI.hparams["batch_size"], num_workers=4)
    pretrained_model_TI.eval()
    plot_conf_matrix(pretrained_model_TI, dataloader_TI)


if __name__ == '__main__':
    print("...........Training Starts............", "\n")
    train_part()