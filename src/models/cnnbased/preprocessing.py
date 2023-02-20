import tensorflow as tf


def scale(data, factor=255):
    data = data.astype("float32") / factor
    return data


def reshape(data):
    data = data.reshape((data.shape[0], 28, 28, 1))
    return data


def onehot_encoding(feature, num_cat=10):
    return tf.keras.utils.to_categorical(feature, num_classes=num_cat)
