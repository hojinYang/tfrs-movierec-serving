from typing import Callable, Dict, Tuple, Text
from recommenders.datasets import Dataset
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from pathlib import Path
from annoy import AnnoyIndex

SAVE_PATH = Path(__file__).resolve().parents[1] / "weights"

class RetrievalModel(tfrs.models.Model):

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
        query_model, candidate_model = network_fn(unique_user_ids = dataset.unique_user_ids, \
                                        unique_item_ids = dataset.unique_movie_ids, **network_args)
        
        self.query_model = query_model
        self.candidate_model = candidate_model
        
        self.cand2emb = dataset.movies.map(lambda features: (features['movieid'], self.candidate_model(features)))

        metrics = tfrs.metrics.FactorizedTopK(dataset.movies.map(self.candidate_model))
        self.task = tfrs.tasks.Retrieval(metrics=metrics)

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        query_embeddings = self.query_model(features)
        candidate_embeddings = self.candidate_model(features)

        # The task computes the loss and the metrics.
        return self.task(query_embeddings, candidate_embeddings, compute_metrics = not training)

    def print_summary(self):
        print(self.query_model.print_summary())
        print(self.candidate_model.print_summary())

    def save_weights(self, save_dir):
        if save_dir is None:
            save_dir = SAVE_PATH
            save_dir.mkdir(parents=True, exist_ok=True)
        self.query_model.save_weights(Path(save_dir) / 'query')
  
        index = AnnoyIndex(self.candidate_model.embedding_dimension, "dot")
        
        for movieid_batch, emb_batch in self.cand2emb.as_numpy_iterator():
            for movieid, emb in zip(movieid_batch, emb_batch):
                index.add_item(movieid, emb)

        index.build(10)
        index.save(str(Path(save_dir) / 'candid.annoy'))


