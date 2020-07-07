import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers

class PairWiseInterationLayer(tf.keras.layers.Layer):
    
    
    def __init__(self,):
        
        super(PairWiseInterationLayer, self).__init__()
        
        
    def call(self, inputs):
        """
        receives an inputs tensor shape: [batch_size, features, embedding_size]
        and returns a tensor shape: [batch_size, (features^2 - features)/2, ebmedding_size]
        """
        
        features = tf.shape(inputs)[1]
        
        muls = []
        for i in range(features):
            for j in range(i, features):
                if i != j:
                    mul = tf.math.multiply(inputs[:, i, :], inputs[:, j, :])
                    muls.append(tf.squeeze(mul))
        
        output = tf.stack(muls)
        
        output = tf.transpose(output, perm=[1, 0 ,2])
        
        return output
    
class AttentionLayer(tf.keras.layers.Layer):
    
    def __init__(self ,attention_factor, reg=0.2):
        super(AttentionLayer, self).__init__()
        
        self.w = tf.keras.layers.Dense(attention_factor, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.af = tf.keras.layers.Dense(1, use_bias=False)
        
    def call(self, x):
        """
        receives a tensor shape [batch_size, (features^2 - features)/2, ebmedding_size]
        returns a tensor shape [batch_size, (features^2 - features)/2, ebmedding_size]
        """
        
        dense = self.w(x) # [batch_size, (features^2 - features)/2, attetion_factor]
        
        attention_scaled = self.af(dense) # [batch_size, (features^2 - features)/2, 1]
        
        attention_score = tf.nn.softmax(attention_scaled, axis=1) # [batch_size, (features^2 - features)/2, 1]
        
        attention_output = attention_score * x
        
        return attention_output
    
    
    
class PoolingLayer(tf.keras.layers.Layer):
    
    def __init__(self, rate=0.1):
        super(PoolingLayer, self).__init__()
        
        self.p = tf.keras.layers.Dense(1)
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training):
        """
        receives a tensor shape [batch_size, (features^2 - features)/2, embedding_size]
        and return a tensor shape [batch_size, 1]
        """
        reduced = tf.math.reduce_sum(x, axis=-2)
        
        output = self.p(reduced)
        
        output = self.dropout(output, training=training)
        
        return output
    
    
class AFMLayer(tf.keras.layers.Layer):
    
    def __init__(self, attention_factor=4, rate=0.1, reg=0.2):
        
        super(AFMLayer, self).__init__()

        self.pairwise = PairWiseInterationLayer()
        self.attention = AttentionLayer(attention_factor, reg)
        self.pooling = PoolingLayer(rate)
        
        
    def call(self, x, training):
        
        pairs = self.pairwise(x)
        
        attention_output = self.attention(pairs)
        
        prediction = self.pooling(attention_output, training)
        
        return prediction


class ADFMLayer(tf.keras.layers.Layer):
    
    def __init__(self, attention_factor=4, rate=0.1, reg=0.2):
        
        super(ADFMLayer, self).__init__()

        self.pairwise = PairWiseInterationLayer()
        self.attention = AttentionLayer(attention_factor, reg)
        self.pooling = PoolingLayer(rate)
        
        
    def call(self, x, deep_x, training):
        
        pairs = self.pairwise(x)
        
        batch_size = tf.shape(deep_x)[0]
        deep_x = tf.reshape(deep_x, shape=[batch_size, 1, -1])
        
        attention_input = tf.concat([pairs, deep_x], axis=1)

        attention_output = self.attention(attention_input)
        
        prediction = self.pooling(attention_output, training)
        
        return prediction
    
    
    
class DeepLayer(tf.keras.layers.Layer):
    
    def __init__(self, dnn=(128,128), rate=0.1, reg=0.2):
        
        super(DeepLayer, self).__init__()

        self.dense_layers = []
        self.dropout = tf.keras.layers.Dropout(rate)
        
        for dense in dnn:
            self.dense_layers.append(tf.keras.layers.Dense(dense, activation='relu'))

        
        self.dense = tf.keras.layers.Dense(1)
            
        
    def call(self, sparse_features, dense_features, training):
        
        """
        recieves
        sparse_features: tensor shape [batch_size, sparse_features, embedding_size]
        dense_features: tensor shape [batch_size, dense_features, 1]
        """
        
        batch_size = tf.shape(sparse_features)[0]

        dense_features = tf.cast(dense_features, tf.float32)
        
        sparse_features_flatten = tf.reshape(sparse_features, shape=[batch_size, -1]) # [batch_size, sparse_features * embedding_size]
        dense_features_flatten = tf.reshape(dense_features, shape=[batch_size, -1]) # [batch_size, dense_features]
        
        x = tf.concat([sparse_features_flatten, dense_features_flatten], axis=-1) # [batch_size, (sparse_features * embedding_size + dense_features)]
        
        for dense in self.dense_layers:
            x = dense(x)
        
        x = self.dropout(x)
        output = self.dense(x)
        
        return output
    

