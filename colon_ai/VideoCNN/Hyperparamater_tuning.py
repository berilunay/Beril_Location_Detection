import optuna
import pytorch_lightning
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization.matplotlib import plot_optimization_history

from colon_ai.VideoCNN.Datamodule import VideoCNNDataModule
from colon_ai.VideoCNN.Model import VideoClassificationLightningModule



def objective(trial):

    # create a trainer
    trainer = pytorch_lightning.Trainer(
        logger=False,
        max_epochs=50,
        gpus=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")]
    )
    # here we sample the hyper params, similar as in our old random search
    trial_hparams = {"weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-3),
                    "batch_size": trial.suggest_int("batch_size", 2, 4),
                    "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
                    "num_workers":1,
                    'gpus': 1
                     }

    module = VideoClassificationLightningModule(trial_hparams)
    data_module = VideoCNNDataModule(trial_hparams)
    trainer.fit(module,data_module)

    return trainer.callback_metrics["val_loss"].item()

if __name__ == '__main__':
    print("...........Testing Hyperparamater Starts............", "\n")
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=25)
    plot_optimization_history(study)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

