import numpy as np
import cv2
import os

mask_path = "data/masks/"
mask_paths = list(os.listdir(mask_path))
mask_paths = np.array([f"{mask_path}{image_path}" for image_path in mask_paths])

image = np.array(cv2.imread("data/Set5/butterfly.png"))


mask = cv2.imread(mask_paths[1])
dim = (image.shape[0], image.shape[1])
mask = cv2.resize(mask, dim)
mask, _, _ = np.array(cv2.split(mask))

masked = cv2.bitwise_or(image, image, mask=mask)

print(mask.shape)


cv2.imshow("img", image)
# cv2.imshow("mask", mask)
cv2.imshow("masked image", masked)
cv2.waitKey()
