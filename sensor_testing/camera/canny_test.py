import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('test.jpg')
edge = cv.GaussianBlur(img, (9, 9), 1)
edge = cv.Canny(edge, 100, 200)

plt.subplot(121), plt.imshow(img, cmap = 'gray')
plt.title('og img'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(edge, cmap = 'gray')
plt.title('canny img'), plt.xticks([]), plt.yticks([])


plt.show()