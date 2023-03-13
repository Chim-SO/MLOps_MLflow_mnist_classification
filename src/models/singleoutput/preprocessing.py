from PIL import Image, ImageChops


def scale(data, factor=255):
    data = data.astype("float32") / factor
    return data


def reshape(data):
    data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    return data


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return im.crop(bbox)
