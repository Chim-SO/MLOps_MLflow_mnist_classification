def scale(data, factor=255):
    data = data.astype("float32") / factor
    return data


def reshape(data):
    data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    return data
