from typing import Callable, Dict, Tuple, Text
from recommenders.datasets import Dataset
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from pathlib import Path

SAVE_PATH = Path(__file__).resolve().parents[1] / "weights"

class RankingModel(tfrs.models.Model):

    def __init__(
        self,
        dataset: Dataset,
        network_fn: Callable,
        network_args: Dict = None
        ):

        super().__init__()
        self._name = f"{self.__class__.__name__}_{network_fn.__name__}"

        if network_args is None:
            network_args = {}

        self.ranking_model: tf.keras.Model = network_fn(
                                            unique_user_ids = dataset.unique_user_ids,
                                            unique_item_ids = dataset.unique_movie_ids, **network_args)
        self.task = tfrs.tasks.Ranking(
          loss = tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        prediction = self.ranking_model(**features)
        return self.task(prediction, features['rating'])

    def call(self, features: Dict[Text, tf.Tensor]):
        return self.ranking_model(**features)

    def print_summary(self):
        print(self.ranking_model.print_summary())

    def save_weights(self, save_dir):
        if save_dir is None:
            save_dir = SAVE_PATH
            save_dir.mkdir(parents=True, exist_ok=True)
            
        self.ranking_model.save_weights(str(Path(save_dir) /'ranking'))

      