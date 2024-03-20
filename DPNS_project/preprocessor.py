import cv2
import numpy as np
import matplotlib as plt

def preprocess(input_image):
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(input_image)
    # preprocessing the image input
    denoised = cv2.fastNlMeansDenoising(input_image)
    ret, tresh = cv2.threshold(denoised, 127, 1, cv2.THRESH_BINARY_INV)
    cropped = crop(tresh)
    # 40x10 image as a flatten array
    flatten_img = cv2.resize(cropped, (40, 10), interpolation=cv2.INTER_AREA).flatten()
    # resize to 400x100
    resized = cv2.resize(cropped, (400, 100), interpolation=cv2.INTER_AREA)
    plt.subplot(222)
    plt.imshow(resized)
    plt.show()
    columns = np.sum(resized, axis=0)  # sum of all columns
    lines = np.sum(resized, axis=1)  # sum of all lines
    h, w = cropped.shape
    aspect = w / h
    return [*flatten_img, *columns, *lines, aspect]


def crop(img):
    points = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(points)
    return img[y: y + h, x: x + w]
