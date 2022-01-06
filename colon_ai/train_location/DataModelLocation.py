import numpy as np
import pandas as pd
import seaborn
import torch
import torchvision
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, wandb
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from pytorch_lightning import LightningModule, Callback
from torch.optim import Adam, SGD
from torch.nn import functional as F
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from torchmetrics.functional import f1
from sklearn.metrics import confusion_matrix
import wandb
from torch.utils.data import DataLoader
from colon_ai.train_location.DataLoader_Location import ColonDataModuleLocation
from colon_ai.train_location.DatasetClass_Location import ColonDatasetLocation
from efficientnet_pytorch import EfficientNet


class ColonDataModelLocation(LightningModule):
    def __init__(self, hparams):
        super(ColonDataModelLocation, self).__init__()
        self.save_hyperparameters(hparams)

        # Network
        #self.network = resnet18(num_classes=3)
        self.network = resnet18(pretrained=True)
        self.num_ftr= self.network.fc.in_features
        self.network.fc=nn.Linear(self.num_ftr,3)
        #self.network = EfficientNet.from_pretrained('efficientnet-b1',num_classes=3)


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



def show_examples(model, datamodule, class_dict=None):
    data_loader=datamodule.test_dataloader()
    for batch_idx, (features, targets) in enumerate(data_loader):
        with torch.no_grad():
            features = features
            targets = targets
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)
        break

    fig, axes = plt.subplots(nrows=5, ncols=5,
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

    print("orig_labels conv:",orig_labels_conv)
    print("predicted labels conv: ",pred_labels_conv)

    conf_matrix =confusion_matrix(orig_labels_conv, pred_labels_conv)
    print("conf_matrix_test: ",conf_matrix)
    df_cm = pd.DataFrame(conf_matrix, index=[i for i in "RML"],
                         columns=[i for i in "RML"])
    plt.figure(figsize=(3, 3))
    seaborn.heatmap(df_cm, annot=True,cmap="OrRd")
    plt.show()

class Datasetview2D_Loc(Callback):
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
    location_dict = {0: 'R', 1: 'M', 2: 'L'}
    hparams = {'weight_decay': 0.00026082233479956905,
               'batch_size': 22,
               'learning_rate': 0.00011694907716441325,
               'num_workers': 4,
               'gpus': 1,
               'test': 1
               }


    model=ColonDataModelLocation(hparams)
    datamodule_colon=ColonDataModuleLocation(hparams)

    #--------------------------------------------------------------------------------------------
    checkpoint_callback = ModelCheckpoint(filename='withoutag--{epoch}-{val_loss:.2f}-{val_acc:.2f}-{train_loss:.2f}-{train_acc:.2f}-{F1_val:.2f}-{F1_train:.2f}',monitor="val_loss", verbose=True)
    trainer=Trainer( max_epochs=15, gpus=hparams["gpus"], logger=WandbLogger(), callbacks=[Datasetview2D_Loc(), checkpoint_callback])
    trainer.fit(model,datamodule_colon)

    trainer.test(datamodule=datamodule_colon)
    #show_examples(model,datamodule_colon,class_dict=location_dict)

    #------------------------------------------------------------------------------
    # Test_Path = "/home/beril/Thesis_Beril/Dataset_preprocess_new/Location_Detection/test_location_labels"
    # val_test_transform = transforms.Compose([
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # checkpoint_model_path_loc = "/home/beril/BerilCodes/ColonAI_LocationDetection/colon_ai/train_location/uncategorized/best_model(11.12)/checkpoints/besthparamstd--epoch=5-val_loss=1.11-val_acc=0.56-train_loss=0.01-train_acc=1.00-F1_val=0.59-F1_train=1.00.ckpt"
    # pretrained_model_loc = ColonDataModelLocation.load_from_checkpoint(checkpoint_path=checkpoint_model_path_loc)
    # pretrained_model_loc.eval()
    # loc_dataset=ColonDatasetLocation(root=Test_Path,transform=val_test_transform)
    # dataloader_colon=DataLoader(loc_dataset, batch_size=pretrained_model_loc.hparams["batch_size"], num_workers=4)
    # #run this part if you want to get the test acc for the loaded models..........................
    # #trainer = Trainer(gpus=pretrained_model_loc.hparams["gpus"])
    # # trainer.test(pretrained_model_loc,dataloaders=dataloader_colon)
    # # pretrained_model_loc.eval()
    # #...........................................................................................
    # plot_conf_matrix(pretrained_model_loc, dataloader_colon)



if __name__ == '__main__':
    print("...........Training Starts............", "\n")
    train_part()