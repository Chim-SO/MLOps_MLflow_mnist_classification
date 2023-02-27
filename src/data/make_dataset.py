import gzip
import os.path
import urllib.request

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


if __name__ == '__main__':
    # Download the MNIST dataset
    train_images_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    train_labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    test_images_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    test_labels_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    raw_data_path = '../../data/raw/'
    urllib.request.urlretrieve(train_images_url, os.path.join(raw_data_path, 'train_images.gz'))
    urllib.request.urlretrieve(train_labels_url, os.path.join(raw_data_path, 'train_labels.gz'))
    urllib.request.urlretrieve(test_images_url, os.path.join(raw_data_path, 'test_images.gz'))
    urllib.request.urlretrieve(test_labels_url, os.path.join(raw_data_path, 'test_labels.gz'))

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
    train_df.to_csv('../../data/processed/train.csv', index=False, header=False)
    test_df.to_csv('../../data/processed/test.csv', index=False, header=False)
