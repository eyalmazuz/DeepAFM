import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from .Layers import PairWiseInterationLayer, AttentionLayer, PoolingLayer, ADFMLayer, DeepLayer
    

class ADFM(tf.keras.Model):
    
    def __init__(self, features_dict, sparse_columns, dense_columns, embedding_size, attention_factor, dnn=(128,128), rate=0.1, reg=0.2):
        
        super(ADFM, self).__init__()
        
        self.fm_embeddings = {}
        self.linear_embeddings = {}

        self.sparse_columns = sparse_columns
        self.dense_columns = dense_columns
        
        for feature_name, feature_size in features_dict.items():
            self.fm_embeddings[feature_name] = tf.keras.layers.Embedding(feature_size+1, embedding_size)
            self.linear_embeddings[feature_name] = tf.keras.layers.Embedding(feature_size+1, 1)
            
        self.adfm = ADFMLayer(attention_factor=attention_factor, rate=rate, reg=reg)
        
        self.dense_layers = []
        
        for dense in dnn:
            self.dense_layers.append(tf.keras.layers.Dense(dense, activation='relu'))

        self.dense_latent = tf.keras.layers.Dense(embedding_size, activation='relu')
    
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
        
        batch_size = tf.shape(fm_inputs)[0]

        dense_inputs = tf.cast(dense_inputs, tf.float32)
        
        sparse_features_flatten = tf.reshape(fm_inputs, shape=[batch_size, -1]) # [batch_size, sparse_features * embedding_size]
        dense_features_flatten = tf.reshape(dense_inputs, shape=[batch_size, -1]) # [batch_size, dense_features]
        
        x = tf.concat([sparse_features_flatten, dense_features_flatten], axis=-1) # [batch_size, (sparse_features * embedding_size + dense_features)]
        
        for dense in self.dense_layers:
            x = dense(x)

        x = self.dense_latent(x)

        adfm_prediction = self.adfm(fm_inputs, x, training)
        
        linear_prediction = tf.math.reduce_sum(linear_inputs, axis=-2)
        
        
        output = tf.nn.sigmoid(adfm_prediction + linear_prediction)
        
        
        return output
