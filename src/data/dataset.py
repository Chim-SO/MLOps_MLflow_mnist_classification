import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    x = df.drop('label', axis=1)
    y = df['label']
    return x.to_numpy(), y.to_numpy()
