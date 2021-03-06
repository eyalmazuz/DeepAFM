import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from .Layers import PairWiseInterationLayer, AttentionLayer, PoolingLayer, AFMLayer, DeepLayer

class AFM(tf.keras.Model):
    
    def __init__(self, features_dict, sparse_columns, embedding_size, attention_factor, rate=0.1, reg=0.2):
        
        super(AFM, self).__init__()
        
        self.fm_embeddings = {}
        self.linear_embeddings = {}
        
        self.sparse_columns = sparse_columns

        for feature_name, feature_size in features_dict.items():
            self.fm_embeddings[feature_name] = tf.keras.layers.Embedding(feature_size+1, embedding_size)
            self.linear_embeddings[feature_name] = tf.keras.layers.Embedding(feature_size+1, 1)
            
        self.afm = AFMLayer(attention_factor, rate, reg)
    
    def call(self, x, training):
        

        sparse_features = {k: np.array(list(v.values())) for k, v in x[self.sparse_columns].to_dict().items()}

        fm_inputs = []
        linear_inputs = []
        for feautre_name, encodings in sparse_features.items():
            fm_inputs.append(self.fm_embeddings[feautre_name](encodings))
            linear_inputs.append(self.linear_embeddings[feautre_name](encodings))

        fm_inputs = tf.stack(fm_inputs, axis=1)
        linear_inputs = tf.stack(linear_inputs, axis=1)

        afm_prediction = self.afm(fm_inputs, training)
        
        linear_prediction = tf.math.reduce_sum(linear_inputs, axis=-2)
        
        output = tf.nn.sigmoid(afm_prediction + linear_prediction)
        
        
        return output