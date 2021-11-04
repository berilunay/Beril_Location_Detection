import optuna
import pytorch_lightning
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization.matplotlib import plot_optimization_history

from colon_ai.VideoCNN_Quality.Datamodule_Quality import VideoCNNDataModuleQuality
from colon_ai.VideoCNN_Quality.model_quality import VideoClassificationLightningModuleQuality


def objective(trial):

    # create a trainer
    trainer = pytorch_lightning.Trainer(
        logger=False,
        max_epochs=60,
        gpus=1,
        #early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_acc"),  # early stopping
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")]
    )

    # here we sample the hyper params, similar as in our old random search
    trial_hparams = {"weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 5e-4),
                    "batch_size": trial.suggest_int("batch_size", 2, 4),
                    "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 5e-4),
                    "num_workers":1,
                    'gpus': 1
                     }

    model = VideoClassificationLightningModuleQuality(trial_hparams)
    data_module = VideoCNNDataModuleQuality(trial_hparams)
    trainer.fit(model,data_module)

    return trainer.callback_metrics["val_acc"].item()

if __name__ == '__main__':
    print("...........Testing Hyperparamater Starts............", "\n")
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=30)
    plot_optimization_history(study)
    #
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

