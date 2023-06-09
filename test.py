import glob

import cv2
path = "fairface/train/*.*"
for file in glob.glob(path):
  image_read = cv2.imread(file)
  width = 680
  height = 480
  dim = (width, height)
  resized = cv2.resize(image_read, dim, interpolation = cv2.INTER_AREA)
  print(file)
