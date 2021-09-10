import optuna
import pytorch_lightning
import torch
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization.matplotlib import plot_optimization_history

from pytorch_lightning import Callback
from pytorch_lightning.utilities import argparse

from VideoCNN.Datamodule import VideoCNNDataModule
from VideoCNN.Model import VideoClassificationLightningModule


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""
    def __init__(self):
        super().__init__()
        self.metrics = []
        print('self.metricsi olusturdum')


    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)
        print('metricsi bastiriyorum:', self.metrics)



# def update_args_(args, params):
#   """updates args in-place"""
#   dargs = vars(args)
#   dargs.update(params)



def objective(trial):
    # as explained above, we'll use this callback to collect the validation accuracies
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--learning_rate", default=1e-4, type=float)
    # parser.add_argument("--weight_decay", default=1e-4, type=float)
    # parser.add_argument("--batch_size", default=4, type=int)
    # args = parser.parse_args()
    #--------------------------------------------------
    #metrics_callback = MetricsCallback()
    SAVE_PATH = "save_run.pt"
    # create a trainer
    trainer = pytorch_lightning.Trainer(
        logger=False,
        max_epochs=100,
        gpus=1,
        #early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_acc"),  # early stopping
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")]
    )

    # here we sample the hyper params, similar as in our old random search
    trial_hparams = {"weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 3e-3),
                    "batch_size": trial.suggest_int("batch_size", 2, 6),
                    "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
                    "num_workers":1,
                    'gpus': 1
                     }

    #--------------------------------------------------------------------------------------
    # params = {"weight_decay": trial.suggest_loguniform("weight_decay", 1e-4, 3e-4),
    #           "batch_size": trial.suggest_int("batch_size", 2, 6),
    #           "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)}
    #
    #update_args_(args, params)
    #-------------------------------------------------------------------------------------
    # create model from these hyper params and train it
    model = VideoClassificationLightningModule(trial_hparams)
    data_module = VideoCNNDataModule(trial_hparams)
    trainer.fit(model,data_module)

    # save model
    #torch.save(model.state_dict(), SAVE_PATH)
    # return validation accuracy from latest model, as that's what we want to minimize by our hyper param search
    #print("test metric:",metrics_callback.metrics)
    #return metrics_callback.metrics[-1]["val_acc"]
    return trainer.callback_metrics["val_acc"].item()

if __name__ == '__main__':
    print("...........Testing Hyperparamater Starts............", "\n")
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=50)
    #plot_optimization_history(study)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

