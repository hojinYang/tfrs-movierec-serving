from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np

def embedding(unique_ids: np.array, embedding_dimension: int) -> tf.keras.Model:
    model = Sequential()
    # Assume training set has no out-of-vocabulary(id) and masked input 
    model.add(tf.keras.layers.experimental.preprocessing.IntegerLookup(num_oov_indices=1, mask_value=None, vocabulary=unique_ids))
    model.add(tf.keras.layers.Embedding(len(unique_ids)+1, embedding_dimension))
    return model
