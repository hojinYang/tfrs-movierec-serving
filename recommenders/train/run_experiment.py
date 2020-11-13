import argparse
import json
import importlib
from typing import Dict
import os
from pathlib import Path
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
# from tensorflow.keras.callbacks import EarlyStopping, Callback

def run_experiment(experiment_config: Dict, save_model: bool=True, use_wandb: bool = True):
    """
    Run a training experiment.
    Parameters
    ----------
    experiment_config (dict)
        Of the form
        {
            "dataset": "Dataset",
            "dataset_args": {
                "batch_size": 128,
                "test_ratio": 0.3
            },
            "model": "RetrievalModel",
            "network": "retrieval_basic_factorization",
            "network_args": {
                "hidden_dim": 64,
            },
            "train_args": {
                "epochs": 10,
                "optimizer": SGD
            }
        }

    """

    print(experiment_config)

    models_module = importlib.import_module("recommenders.models")
    model_class_ = getattr(models_module, experiment_config["model"])

    networks_module = importlib.import_module("recommenders.networks")
    network_fn_ = getattr(networks_module, experiment_config["network"])
    network_args = experiment_config.get("network_args", {})

    datasets_module = importlib.import_module("recommenders.datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    dataset_args = experiment_config.get("dataset_args", {})
    dataset = dataset_class_(**dataset_args)
    dataset.load_or_generate_data()
    
    model = model_class_(
        dataset=dataset, network_fn=network_fn_, network_args=network_args
    )
    
    if use_wandb:
        wandb.init(config=experiment_config)
    
    callbacks = list()
    callbacks.append(WandbCallback())
    optimizer = _get_optimizer(experiment_config["train_args"]["optimizer"])
    model.compile(optimizer=optimizer(learning_rate=experiment_config["train_args"]["learning_rate"]))

    model.fit(dataset.train, 
            epochs=experiment_config["train_args"]["epochs"], 
            validation_data=dataset.test,
            validation_freq=20,
            callbacks=callbacks)

    model.print_summary()
    
    if save_model:
        if use_wandb:
            model.save_weights(wandb.run.dir)
        else:
            model.save_weights()

def _get_optimizer(name: str = 'SGD'):
    if name == 'Adagrad':
        return tf.keras.optimizers.Adagrad
    elif name == 'Adam':
        return tf.keras.optimizers.Adam
    elif name == 'RMSprop':
        return tf.keras.optimizers.RMSprop
    elif name == 'Nadam':
        return tf.keras.optimizers.Nadam
    elif name == 'SGD':
        return tf.keras.optimizers.SGD

def main():
    """
    d=  {
            "dataset": "Dataset",
            "dataset_args": {
                "batch_size": 512,
                "test_fraction": 0.3
            },
            "model": "RetrievalModel",
            "network": "retrieval_basic_factorization",
            "network_args": {
                "embedding_dimension": 32,
            },
            "train_args": {
                "epochs": 1,
                "learning_rate": 0.01,
                "optimizer": "SGD"

            }
        }
    """
    d=  {
            "dataset": "Dataset",
            "dataset_args": {
                "batch_size": 512,
                "test_fraction": 0.3
            },
            "model": "RankingModel",
            "network": "ranking_neural_cf",
            "network_args": {
                "embedding_dimension": 32,
                "hidden_layers": [64, 32],
                "dropout_amount": 0.2,
                "regularization_weight": 1e-2
            },
            "train_args": {
                "epochs": 1,
                "learning_rate": 0.01,
                "optimizer": "SGD"
            }
        }

    run_experiment(d, True, True)

if __name__ == "__main__":
    main()
