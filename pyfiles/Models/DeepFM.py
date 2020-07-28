import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
# from . import Layers
from .Layers import FMLayer, DeepLayer
    
class DeepFM(tf.keras.Model):
    
    def __init__(self, features_dict, sparse_columns, dense_columns, embedding_size, dnn=(128,128), rate=0.1, reg=0.2):
        
        super(DeepFM, self).__init__()
        
        self.fm_embeddings = {}
        self.linear_embeddings = {}
        
        self.sparse_columns = sparse_columns
        self.dense_columns = dense_columns

        for feature_name, feature_size in sorted(features_dict.items()):
            self.fm_embeddings[feature_name] = tf.keras.layers.Embedding(feature_size+1, embedding_size, embeddings_regularizer=regularizers.l2(reg))
            self.linear_embeddings[feature_name] = tf.keras.layers.Embedding(feature_size+1, 1)
            
        self.fm = FMLayer()
        
        self.deep = DeepLayer(dnn=dnn, rate=rate, reg=reg)
    
    def call(self, x, training):
        
        
        sparse_features = {k: np.array(list(v.values())) for k, v in x[self.sparse_columns].to_dict().items()}
        dense_features = {k: np.array(list(v.values())) for k, v in x[self.dense_columns].to_dict().items()}

        fm_inputs = []
        linear_inputs = []
        for feautre_name, encodings in sparse_features.items():
            fm_inputs.append(self.fm_embeddings[feautre_name](encodings))
            linear_inputs.append(self.linear_embeddings[feautre_name](encodings))

            
        fm_inputs = tf.stack(fm_inputs, axis=1)
        linear_inputs = tf.stack(linear_inputs, axis=1)
        if dense_features:
            dense_inputs = tf.stack(list(dense_features.values()), axis=1)
        else:
            dense_inputs = tf.stack([])
        
        fm_prediction = self.fm(fm_inputs)
        
        linear_prediction = tf.math.reduce_sum(linear_inputs, axis=-2)
        
        deep_prediction = self.deep(fm_inputs, dense_inputs, training)
        
        output = tf.nn.sigmoid(deep_prediction + fm_prediction + linear_prediction)
        
        
        return output