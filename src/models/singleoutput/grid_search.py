import mlflow

from src.models.singleoutput.train import train


def grid_search(data_path, params):
    for n_l in params['n_layers']:
        for n_u in params['units']:
            for ac in params['activation_function']:
                for b in params['batch_size']:
                    with mlflow.start_run():
                        train(data_path, n_l, n_u, ac, params['loss'], params['metric'], params['epochs'], b)


if __name__ == '__main__':
    data_path = "../../../data/processed/"
    mlflow.set_tracking_uri('file:../../../models/mlruns')
    mlflow.set_experiment('singleOutput')
    params = {
        'n_layers': [2, 4, 6],
        'units': [16, 64, 256],
        'activation_function': ['relu', 'sigmoid'],
        'batch_size': [64, 256],
        'loss': 'mse',
        'metric': 'mae',
        'epochs': 300
    }
    grid_search(data_path, params)
