import mlflow
import numpy as np
from PIL import Image, ImageChops

from src.models.singleoutput.preprocessing import scale, reshape
import matplotlib.pyplot as plt

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return im.crop(bbox)


if __name__ == '__main__':
    # Load image:
    im = Image.open("../../../data/external/test/5_0.png").convert('L')
    # im.show()

    # Trim image:
    im_trim = trim(im)
    # im_trim.show()

    # Resize image:
    im_res = im_trim.resize((28, 28))
    # im_res.show()

    # Image preprocessing:
    x = 1 - np.array(im_res)
    x = scale(x)
    x = np.expand_dims(x, 0)
    print(x.shape)
    x = reshape(x)
    print(x.shape)

    # Load model:
    mlflow.set_tracking_uri('file:../../../models/mlruns')
    logged_model = 'runs:/70f19bb490fe413482d066d36fe62d80/models'
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
    ax[3].imshow(x.reshape((28,28)), cmap='gray')
    ax[3].set_title('Preprocessed')
    plt.show()
