from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras.models import Sequential


def create_model(input_shape, n_layers=5, n_units=224, activation='sigmoid'):
    # Create model:
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for _ in range(n_layers):
        model.add(Dense(n_units, activation=activation))
    model.add(Dense(1, activation='relu'))
    return model
