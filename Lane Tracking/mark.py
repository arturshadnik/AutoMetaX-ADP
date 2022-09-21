import numpy as np
import cv2
mask=np.zeros((4, 3))
cv2.fillPoly(mask, [(1,1),(1,2), (2,1), (2,2)], 255)