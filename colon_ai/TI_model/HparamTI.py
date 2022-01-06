import optuna
import pytorch_lightning
from optuna.integration import PyTorchLightningPruningCallback

from colon_ai.TI_model.DataLoaderTI import ColonDataModuleTI
from colon_ai.TI_model.DatamodelTI import ColonDataModel_TI


def objective(trial):

    # create a trainer
    trainer = pytorch_lightning.Trainer(
        logger=False,
        max_epochs=5,
        gpus=1,
        #early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_acc"),  # early stopping
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")]
    )
    SAVE_PATH="saved_model.pth"
    # here we sample the hyper params, similar as in our old random search
    trial_hparams = {"weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-3),
                    "batch_size": trial.suggest_int("batch_size", 16, 64),
                    "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-3),
                    "num_workers":4,
                    'gpus': 1
                     }

    #-------------------------------------------------------------------------------------
    # create model from these hyper params and train it
    model = ColonDataModel_TI(trial_hparams)
    datamodule_colon = ColonDataModuleTI(trial_hparams)
    trainer.fit(model,datamodule_colon)

    return trainer.callback_metrics["val_loss"].item()

if __name__ == '__main__':
    print("...........Testing Hyperparamater Starts............", "\n")
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=7)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

