import matplotlib.pyplot as plt
import mlflow
import numpy as np
from PIL import Image

from src.models.singleoutput.preprocessing import scale, reshape, trim, add_border

if __name__ == '__main__':
    # Load image:
    im = Image.open("../../../data/external/test/0_0.png").convert('L')
    # im.show()

    # Trim image:
    im_trim = trim(im)
    # im_trim.show()

    # Resize image:
    b = 4
    im_res = im_trim.resize((28 - 2 * b, 28 - 2 * b))
    im_res = add_border(im_res, b)

    # Image preprocessing:
    x = 1 - np.array(im_res)
    x = scale(x)
    x = np.expand_dims(x, 0)
    print(x.shape)
    x = reshape(x)
    print(x.shape)

    # Load model:
    mlflow.set_tracking_uri('file:../../../mlruns')
    logged_model = 'runs:/5bc3e2f16e254125a966090afccb085d/models'
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict:
    y = loaded_model.predict(x)

    # Print:
    fig, ax = plt.subplots(1, 4)
    fig.suptitle(f'Predicted: {round(y[0][0])}')
    ax[0].imshow(im, cmap='gray')
    ax[0].set_title('Input')
    ax[1].imshow(im_trim, cmap='gray')
    ax[1].set_title('Trimmed')
    ax[2].imshow(im_res, cmap='gray')
    ax[2].set_title('Resized')
    ax[3].imshow(x.reshape((28, 28)), cmap='gray')
    ax[3].set_title('Preprocessed')
    plt.show()
