"""W&B Sweep Functionality."""
import os
import signal
import subprocess
import sys
import json
import run_experiment
from typing import Tuple
from ast import literal_eval


#You need to select appropriate default config(retrieval or ranking)
DEFAULT_CONFIG = {
            "dataset": "Dataset",
            "dataset_args": {
                "batch_size": 2048,
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
                "epochs": 60,
                "learning_rate": 0.01,
                "optimizer": "SGD"
            }
        }
'''

DEFAULT_CONFIG = {
            "dataset": "Dataset",
            "dataset_args": {
                "batch_size": 4056,
                "test_fraction": 0.3
            },
            "model": "RetrievalModel",
            "network": "retrieval_basic_factorization",
            "network_args": {
                "embedding_dimension": 32,
            },
            "train_args": {
                "epochs": 60,
                "learning_rate": 0.01,
                "optimizer": "SGD"
            }
        }
'''
def args_to_json(default_config: dict, preserve_args: tuple = ("save_model")) -> Tuple[dict, list]:
    """Convert command line arguments to nested config values
    i.e. run_sweep.py --dataset_args.foo=1.7
    {
        "dataset_args": {
            "foo": 1.7
        }
    }
    """
    args = []
    config = default_config.copy()
    key, val = None, None
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=")
        elif key:
            val = arg
        else:
            key = arg
        if key and val:
            parsed_key = key.lstrip("-").split(".")
            if parsed_key[0] in preserve_args:
                args.append("--{}={}".format(parsed_key[0], val))
            else:
                nested = config
                for level in parsed_key[:-1]:
                    nested[level] = config.get(level, {})
                    nested = nested[level]
                try:
                    # Convert numerics to floats / ints
                    val = literal_eval(val)
                except ValueError:
                    pass
                nested[parsed_key[-1]] = val
            key, val = None, None
    return config, args


def main():
    config, args = args_to_json(DEFAULT_CONFIG)
    run_experiment.run_experiment(config)


if __name__ == "__main__":
    main()