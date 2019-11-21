from cv2 import cv2
import numpy as np 
import cv2_helpers

img = cv2.imread("pictures/rose.jpeg",0)

color_image = cv2_helpers.get_colored(img)

cv2.imshow("color image", color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()