from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
from .embedding import embedding

class Embedding(tf.keras.Model):
    def __init__(self, unique_ids, embedding_dimension, id_type):
        super().__init__()
        
        self.embeddings: tf.keras.Model = embedding(unique_ids, embedding_dimension)
        self.id_type = id_type
        self._embedding_dimension = embedding_dimension
        
    def call(self, features):
        # FIXME: setting argment **features brings error when saving the model..
        return self.embeddings(features[self.id_type])

    def print_summary(self):
        print(self.summary())

    @property
    def embedding_dimension(self):
        return self._embedding_dimension

def retrieval_basic_factorization(
    unique_user_ids: np.array, 
    unique_item_ids: np.array, 
    embedding_dimension: int = 32,
    **unused) -> tf.keras.Model:

    user_embedding = Embedding(unique_user_ids, embedding_dimension, 'userid')
    item_embedding = Embedding(unique_item_ids, embedding_dimension, 'movieid')

    return user_embedding, item_embedding
