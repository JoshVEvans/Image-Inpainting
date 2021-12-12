import tensorflow as tf


import numpy as np
import random
import cv2


# This method returns the batch of images using multiprocessing for faster interpolation
def load_multiprocessing(multiprocessing_pool, data):
    batch = multiprocessing_pool.map(get_set, data)

    X = []
    y = []

    for iteration in batch:
        X.append(iteration[0])
        y.append(iteration[1])

    X = np.array(X) / 255
    y = np.array(y) / 255

    return X, y


# This method returns the batch of images of length batch_size
def load(data):
    X = []
    y = []

    for image_path in data:
        a, b = get_set(data)

        X.append(a)
        y.append(b)

    X = np.array(X) / 255
    y = np.array(y) / 255

    return X, y


# This method returns a set of X, y images.
def get_set(data):
    dim = 96
    # Read Image and Mask from Path
    image = cv2.imread(data[0])
    mask = np.array(cv2.imread(data[1]))

    # Crop / Resize Sample
    temp_dim = image.shape

    if temp_dim[0] >= dim and temp_dim[1] >= dim:
        # Crop Image
        image = np.array(tf.image.random_crop(image, size=(dim, dim, 3)))
    else:
        # Resize Image
        image = cv2.resize(image, (dim, dim))

    ### Data Augmentation ###
    # Horizontal Flipping
    if random.choice([True, False]):
        image = cv2.flip(image, 1)
    # Vertical Flipping
    if random.choice([True, False]):
        image = cv2.flip(image, 0)
    # Rotation
    if random.choice([True, False]):
        temp = random.choice([0, 1, 2])
        image = cv2.rotate(image, temp)

    # Store y
    y = image.copy()

    mask, _, _ = np.array(cv2.split(mask))
    mask = cv2.resize(mask, (dim, dim))
    image = cv2.bitwise_or(image, image, mask=mask)

    ### Image Interpolation ###
    # x1 = random.randint(0, dim)
    # x2 = random.randint(x1, dim)
    # y1 = random.randint(0, dim)
    # y2 = random.randint(y1, dim)
    # image[x1:x2, y1:y2] = 255

    # Store X
    X = image

    # cv2.imshow("img", np.concatenate((X, y), axis=1))
    # cv2.waitKey()

    return X, y
