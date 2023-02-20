import argparse
import itertools

import mlflow
import yaml

from src.models.singleoutput.train import train


def grid_search(data_path, params):
    for n_l in params['n_layers']:
        for n_u in params['units']:
            for ac in params['activation_function']:
                for b in params['batch_size']:
                    train(data_path, n_l, n_u, ac, params['loss'][0], params['metric'][0], params['epochs'][0], b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning of a Keras model with single output for MNIST classification")
    parser.add_argument("--config-file", "-c", type=str, default='../../../configs/singleoutput.yaml')
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Get parameters:
    data_path = config['data']['dataset_path']

    # MLflow setup
    mlflow.set_tracking_uri(config['mlflow']['mlruns_path'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Compute all possible combinations:
    params = config['hyperparameter_tuning']
    keys, values = zip(*params.items())
    runs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Train:
    for run in runs:
        with mlflow.start_run():
            train(data_path, run['num_layers'], run['num_units'], run['activation_function'],
                  run['loss_function'], run['metric'], run['num_epochs'], run['batch_size'])
