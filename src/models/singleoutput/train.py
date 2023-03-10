import argparse
import os.path
import random

import mlflow.keras
import numpy as np
import tensorflow as tf
import yaml
from numpy.random import seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.data.dataloader import load_data
from src.models.singleoutput.model import create_model
from src.models.singleoutput.preprocessing import scale

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


def train(data_path, n_layers, n_units, activation_function, loss, metric, epochs, batch_size):
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
    mlflow.tensorflow.log_model(model, 'models')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Keras model with single output for MNIST classification")
    parser.add_argument("--config-file", "-c", type=str, default='../../../configs/singleoutput_0.yaml')
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Get parameters:
    data_path = config['data']['dataset_path']

    # MLflow setup
    # mlflow.set_tracking_uri(config['mlflow']['mlruns_path'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    with mlflow.start_run():
        train(data_path, config['model']['num_layers'], config['model']['num_units'],
              config['model']['activation_function'], config['training']['loss_function'], config['training']['metric'],
              config['training']['num_epochs'],
              config['training']['batch_size'])
