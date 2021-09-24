import optuna
import pytorch_lightning
from optuna.integration import PyTorchLightningPruningCallback

from colon_ai.train_location.DataLoader_Location import ColonDataModuleLocation
from colon_ai.train_location.DataModelLocation import ColonDataModelLocation


def objective(trial):

    # create a trainer
    trainer = pytorch_lightning.Trainer(
        logger=False,
        max_epochs=55,
        gpus=1,
        #early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_acc"),  # early stopping
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")]
    )
    SAVE_PATH="saved_model.pth"
    # here we sample the hyper params, similar as in our old random search
    trial_hparams = {"weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 3e-5),
                    "batch_size": trial.suggest_int("batch_size", 100, 128),
                    "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-3),
                    "num_workers":4,
                    'gpus': 1
                     }

    #-------------------------------------------------------------------------------------
    # create model from these hyper params and train it
    model = ColonDataModelLocation(trial_hparams)
    datamodule_colon = ColonDataModuleLocation(trial_hparams)
    trainer.fit(model,datamodule_colon)

    # save model
    #torch.save(model.state_dict(), SAVE_PATH)
    # return validation accuracy from latest model, as that's what we want to minimize by our hyper param search
    return trainer.callback_metrics["val_acc"].item()

if __name__ == '__main__':
    print("...........Testing Hyperparamater Starts............", "\n")
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

