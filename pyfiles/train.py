import random
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys

import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from absl import app
from absl import flags

from Models import AFM, DeepAFM, ADFM

FLAGS = flags.FLAGS

flags.DEFINE_integer('embedding_size', 256, 'model embedding size', lower_bound=1)
flags.DEFINE_integer('epochs', 5, 'number of epochs to train', lower_bound=1)
flags.DEFINE_integer('batch_size', 256, 'batch size for the epoch', lower_bound=1)
flags.DEFINE_integer('attention_factor', 16, 'model attention facotrs', lower_bound=1)

flags.DEFINE_float('validation_size', 0.2, 'batch size for the epoch', lower_bound=0.001)
flags.DEFINE_float('test_size', 0.1, 'model attention facotrs', lower_bound=0.001)
flags.DEFINE_float('learning_rate', 0.1, 'model learning rate', lower_bound=0)
flags.DEFINE_float('dropout', 0.1, 'dropout rate', lower_bound=0)
flags.DEFINE_float('regularization', 0.1, 'regularization rate', lower_bound=0)

flags.DEFINE_string('model', 'DeepAFM', 'model to train')
flags.DEFINE_string('save_path', './', 'location to save the model')
flags.DEFINE_string('dataset_path', None, 'dataset path to train on')

flags.DEFINE_list('dnn', [128, 128], 'how many nodes in each layer of the dnn component')

flags.DEFINE_bool('eval', True, 'eval model on test set')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)


def load_data(path):
    df = pd.read_csv(path)

    labels = df['label']
    df.drop(columns=['label'], inplace=True)

    return df, labels

def split_data(df, labels, validation_size, test_size):
    x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def preprocess_data(df):


    sparse_columns = df.select_dtypes(include=['int64', 'object']).columns

    dense_columns = df.select_dtypes(include=['float64']).columns


    sparse_encoders = {}
    for column in sparse_columns:
        sparse_encoders[column] = LabelEncoder()
        df[column] = sparse_encoders[column].fit_transform(df[column].values)
    
    dense_encoders = {}
    for column in dense_columns:
        dense_encoders[column] = StandardScaler()
        df[column] = dense_encoders[column].fit_transform(df[column].values.reshape(-1,1))

    features = df.nunique().to_dict()

    return df, features, sparse_columns, dense_columns

def loss_function(y_true, y_pred, loss_object):
    
    loss = loss_object(y_true=y_true, y_pred=y_pred)
    
    rmse = tf.math.sqrt(loss)
    return rmse


def get_model(features, model_name, embedding_size, dnn_size, attention_factor, regularization, dropout, sparse_columns, dense_columns):

    if model_name == "AFM":
        model = AFM.AFM(features, sparse_columns, embedding_size=embedding_size, attention_factor=attention_factor, rate=dropout, reg=regularization)

    elif model_name == "DeepAFM":
        model = DeepAFM.DeepAFM(features, sparse_columns, dense_columns, embedding_size=embedding_size, attention_factor=attention_factor, rate=dropout, reg=regularization, dnn=dnn_size)

    elif model_name == "ADFM":
        model = ADFM.ADFM(features, sparse_columns, dense_columns, embedding_size=embedding_size, attention_factor=attention_factor, rate=dropout, reg=regularization, dnn=dnn_size)
    
    return model


def train_step(input, target, model, optimizer, loss_object):

    with tf.GradientTape() as tape:
        
        predictions  = model(input, True)
        
        loss = loss_function(y_true=target, y_pred=predictions, loss_object=loss_object)
                
    gradients = tape.gradient(loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return predictions


def train(x_train, x_val, y_train, y_val, features, model):

    EPOCHS = FLAGS.epochs
    BATCH_SIZE = FLAGS.batch_size
    STEPS = x_train.shape[0] // BATCH_SIZE


    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=FLAGS.learning_rate)

    train_entropy = tf.keras.metrics.BinaryCrossentropy(name='train_entropy')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_auc = tf.keras.metrics.AUC(name='train_auc')

    val_entropy = tf.keras.metrics.BinaryCrossentropy(name='val_entropy')
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')
    val_auc = tf.keras.metrics.AUC(name='val_auc')

    if FLAGS.save_path: 
        checkpoint_path = FLAGS.save_path

        ckpt = tf.train.Checkpoint(model=model,
                                optimizer=optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

    for epoch in range(EPOCHS):
        start = time.time()

        train_entropy.reset_states()
        train_accuracy.reset_states()
        train_auc.reset_states()

        
        val_entropy.reset_states()
        val_accuracy.reset_states()
        val_auc.reset_states()

        for batch in range(STEPS):
            
            sample = x_train.sample(n=BATCH_SIZE)
            indexs = sample.index
            y = y_train[indexs].values.reshape((-1,1))
            
            predictions = train_step(sample, y, model, optimizer, loss_object)

            train_entropy(y_true=y, y_pred=predictions)
            train_accuracy(y_true=y, y_pred=predictions)
            train_auc(y_true=y, y_pred=predictions)

            if batch % 400 == 0:
                print ('Epoch {} Batch {} Binray Crossentropy {:.4f} Accuracy {:.4f} AUC {:.4f}'.format(
                epoch + 1, batch, train_entropy.result(), train_accuracy.result(), train_auc.result()))

        for batch in range(x_val.shape[0] // BATCH_SIZE):

            sample = x_val.sample(n=BATCH_SIZE)
            indexs = sample.index
            y = y_val[indexs].values.reshape((-1,1))
            val_predictions = model(sample, False)

            val_entropy(y_true=y, y_pred=val_predictions)
            val_accuracy(y_true=y, y_pred=val_predictions)
            val_auc(y_true=y, y_pred=val_predictions)

        print()
        print('Validation Binray Crossentropy {:.4f} Accuracy {:.4f} AUC {:.4f}'.format(
        val_entropy.result(), val_accuracy.result(), val_auc.result()))
        print()

        
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))

            print ('Epoch {} Crossentropy {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                        train_entropy.result(), 
                                                        train_accuracy.result()))

            print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def predict_test(model, x_test, y_test):

    BATCH_SIZE = FLAGS.batch_size
    STEPS = x_test.shape[0] // BATCH_SIZE

    test_entropy = tf.keras.metrics.BinaryCrossentropy(name='test_entropy')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
    test_auc = tf.keras.metrics.AUC(name='test_auc')

    for batch in range(STEPS):

            sample = x_test.sample(n=BATCH_SIZE)
            indexs = sample.index
            y = y_test[indexs].values.reshape((-1,1))
            test_predictions = model(sample, False)

            test_entropy(y_true=y, y_pred=test_predictions)
            test_accuracy(y_true=y, y_pred=test_predictions)
            test_auc(y_true=y, y_pred=test_predictions)

    print('Test Binray Crossentropy {:.4f} Accuracy {:.4f} AUC {:.4f}'.format(
        test_entropy.result(), test_accuracy.result(), test_auc.result()))

def main(argv):
  
    df, labels = load_data(FLAGS.dataset_path)
    df, features, sparse_columns, dense_columns = preprocess_data(df)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(df, labels, FLAGS.validation_size, FLAGS.test_size)

    model = get_model(features, model_name=FLAGS.model, embedding_size=FLAGS.embedding_size, dnn_size=FLAGS.dnn,
                    attention_factor=FLAGS.attention_factor, regularization=FLAGS.regularization, dropout=FLAGS.dropout,
                    sparse_columns=sparse_columns, dense_columns=dense_columns)

    train(x_train, x_val, y_train, y_val, features, model)

    if FLAGS.eval:
        predict_test(model, x_test, y_test)

if __name__ == "__main__":

    app.run(main)

