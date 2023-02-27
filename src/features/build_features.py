import gzip
import os

import numpy as np
import pandas as pd


# Load the dataset
def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28)


def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def preprocess(raw_data_path, preprocessed_data_path):
    # Load images and labels:
    train_images = load_images(os.path.join(raw_data_path, 'train_images.gz'))
    train_labels = load_labels(os.path.join(raw_data_path, 'train_labels.gz'))
    test_images = load_images(os.path.join(raw_data_path, 'test_images.gz'))
    test_labels = load_labels(os.path.join(raw_data_path, 'test_labels.gz'))

    # Concatenate the images and labels
    train_data = np.concatenate((train_images, train_labels[:, np.newaxis]), axis=1)
    test_data = np.concatenate((test_images, test_labels[:, np.newaxis]), axis=1)

    # Convert to a Pandas DataFrame
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Save to CSV files
    train_df.to_csv(os.path.join(preprocessed_data_path, 'train.csv'), index=False, header=False)
    test_df.to_csv(os.path.join(preprocessed_data_path, 'test.csv'), index=False, header=False)
