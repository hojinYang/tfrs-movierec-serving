
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from .embedding import embedding

class RankingNeuralCF(tf.keras.Model):

    def __init__(self, 
                unique_user_ids, 
                unique_movie_ids, 
                embedding_dimension, 
                hidden_layers,
                dropout_amount, 
                regularization_weight):
        super().__init__()
        
        self.query_embeddings: tf.keras.Model = embedding(unique_user_ids, embedding_dimension)
        self.candidate_embeddings: tf.keras.Model = embedding(unique_movie_ids, embedding_dimension)
        self.model = Sequential()
        for layer_dim in hidden_layers:
            self.model.add(Dense(layer_dim, 
                                    activation="relu",
                                    kernel_regularizer=regularizers.l2(regularization_weight),
                                    bias_regularizer=regularizers.l2(regularization_weight)))
            self.model.add(Dropout(dropout_amount)) 
        self.model.add(Dense(1))

    
    def call(self, userid, movieid, **unused):
        query_embedding = self.query_embeddings(userid)
        candidate_embedding = self.candidate_embeddings(movieid)
        return self.model(tf.concat(values=[query_embedding, candidate_embedding],axis=1))

    def print_summary(self):
        print(self.query_embeddings.summary())
        print(self.candidate_embeddings.summary())
        print(self.model.summary())

def ranking_neural_cf(
    unique_user_ids: np.array, 
    unique_item_ids: np.array, 
    embedding_dimension: int = 32, 
    hidden_layers: list = [64, 32],
    dropout_amount: float = 0.2,
    regularization_weight: float = 0., 
    **unused) -> tf.keras.Model:

    return RankingNeuralCF(unique_user_ids, unique_item_ids, embedding_dimension, 
                            hidden_layers, dropout_amount, regularization_weight)