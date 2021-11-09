import cv2
import numpy as np
import random
import tensorflow as tf
import os

# Hides tensorflow outputs and warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Prevents complete memory allocation of gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


image = np.array(cv2.imread("Inpainting/baboon.png"))

dim = image.shape
minX = dim[0] // 10
minY = dim[1] // 10

# image = np.array(tf.image.random_crop(image, size=(dim, dim, 3)))

shape = image.shape
print(shape)


x1 = random.randint(0, dim[0] - minX)
x1 = 5
x2 = random.randint(x1, dim[0])
while x2 - x1 > 100:
    x2 = random.randint(x1, dim[0])

y1 = random.randint(0, dim[1] - minY)
y1 = 5
y2 = random.randint(y1, dim[1])
while y2 - y1 > 100:
    y2 = random.randint(y1, dim[1])


x = [x1, x2]
y = [y1, y2]
image[x1:x2, y1:y2] = 255
print(f"x: {x}, dim: {x2 - x1}")
print(f"y: {y}, dim: {y2 - y1}")

cv2.imwrite("img.png", image)
