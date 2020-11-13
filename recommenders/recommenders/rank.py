from pathlib import Path
import tensorflow as tf
import importlib
import numpy as np

DEFAULT_CONFIG =  {
        "dataset": "Dataset",
        "model": "RankingModel",
        "network": "ranking_neural_cf",
        "network_args": {
            "embedding_dimension": 32,
            "hidden_layers": [32, 16]
        }
}

class RankingModel:
    def __init__(self, weight_dir, config=DEFAULT_CONFIG):

        networks_module = importlib.import_module("recommenders.networks")
        network_fn_ = getattr(networks_module, config["network"])
        network_args = config.get("network_args", {})

        datasets_module = importlib.import_module("recommenders.datasets")
        dataset_class_ = getattr(datasets_module, config["dataset"])
        dataset_args = config.get("dataset_args", {})
        dataset = dataset_class_(**dataset_args)
        dataset.load_or_generate_data(update_to_latest_db=False)

        self.model = network_fn_(dataset.unique_user_ids, dataset.unique_movie_ids, **network_args)
        self.model.load_weights(weight_dir / 'ranking')

    def predict(self, features):
        return np.round(self.model(**features).numpy().squeeze(),1)
