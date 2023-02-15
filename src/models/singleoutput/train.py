import argparse
import os.path
import random

import mlflow.keras
import numpy as np
import tensorflow as tf
from numpy.random import seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from model import create_model
from preprocessing import scale
from src.data.dataset import load_data

seed(1)
tf.random.set_seed(1)
tf.config.experimental.enable_op_determinism()
random.seed(2)


def eval_metrics(actual, pred):
    acc = round(accuracy_score(actual, pred, normalize=True) * 100, 2)
    precision = round(precision_score(actual, pred, average='macro') * 100, 2)
    recall = round(recall_score(actual, pred, average='macro') * 100, 2)
    f1 = round(f1_score(actual, pred, average='macro') * 100, 2)
    return acc, precision, recall, f1


def train(data_path, n_layers, n_units, activation_function, loss, metric, epochs, batch_size, name='singleOutput'):

    # Log parameters:
    mlflow.log_param("n_layers", n_layers)
    mlflow.log_param("units", n_units)
    mlflow.log_param("activation_function", activation_function)
    mlflow.log_param("loss", loss)
    mlflow.log_param("metric", metric)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    # Read dataset:
    x_train, y_train = load_data(os.path.join(data_path, 'train.csv'))
    # Scaling:
    x_train = scale(x_train)
    # Split dataset:
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    # Create model:
    model = create_model((x_train.shape[1],), n_layers=n_layers, n_units=n_units,
                         activation=activation_function)

    # Train:
    class LogMetricsCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mlflow.log_metric('training_loss', logs['loss'], epoch)
            mlflow.log_metric(f'training_{metric}', logs[metric], epoch)
            mlflow.log_metric('val_loss', logs['val_loss'], epoch)
            mlflow.log_metric(f'val_{metric}', logs[f'val_{metric}'], epoch)

    model.compile(loss=loss, optimizer='adam', metrics=[metric])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                        validation_data=(x_val, y_val), callbacks=[LogMetricsCallback()])

    # Evaluation
    x_test, y_test = load_data(os.path.join(data_path, 'test.csv'))
    # Scaling:
    x_test = scale(x_test)

    # evaluate and log:
    test_loss, test_metric = model.evaluate(x_test, y_test, verbose=2)
    mlflow.log_metric('test_loss', test_loss)
    mlflow.log_metric('test_loss', test_loss)

    # Other metric evaluation:
    y_pred = np.rint(np.clip(model.predict(x_test), 0, 9))

    acc, precision, recall, f1 = eval_metrics(y_test, y_pred)
    mlflow.log_metric('test_acc', acc)
    mlflow.log_metric('test_precision', precision)
    mlflow.log_metric('test_recall', recall)
    mlflow.log_metric('test_f1', f1)

    # Log model:
    mlflow.keras.log_model(model, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Keras model with single output for MNIST classification")
    parser.add_argument("--n-layers", "-n", type=int, default=5)
    parser.add_argument("--units", "-u", type=int, default=224)
    parser.add_argument("--activation-function", "-a", type=str, default='sigmoid')
    parser.add_argument("--loss", "-l", type=str, default='mae')
    parser.add_argument("--metric", "-m", type=str, default='mse')
    parser.add_argument("--epochs", "-e", type=int, default=200)
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    args = parser.parse_args()

    data_path = "../../../data/processed/"
    train(data_path, args.n_layers, args.units, args.activation_function, args.loss, args.metric, args.epochs,
          args.batch_size)
