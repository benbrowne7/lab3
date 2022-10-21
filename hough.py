import cv2
import numpy as np

gradient_magnitude = cv2.imread("gradientmag.jpg", -1 );
gradient_direction = cv2.imread("gradientdirection.jpg", -1 );

def hough(gradient_magnitude, gradient_direction, T):
  threshold_image = threshold(gradient_magnitude, T)
  d1 = threshold_image.shape[0]
  print(threshold_image.shape[0])
  print(threshold_image.shape[1])
  d2 = threshold_image.shape[1]
  rs = np.linspace(1, 75, 75, dtype=int)
  hough3D = np.zeros([d1, d2, len(rs)], dtype=np.float32)

  for y in range(0, threshold_image.shape[0]):
    for x in range(0, threshold_image.shape[1]):
      if threshold_image[y,x] == 255:
        for k in rs:
          x0 = int(y + k * np.cos(gradient_direction[y,x])) 
          y0 = int(x + k * np.cos(gradient_direction[y,x]))
          if x0 <= d1 - k and y0 <= d2 - k:
            hough3D[x0,y0,k-1] = hough3D[x0,y0,k-1] + 1
          x0 = int(y - k * np.cos(gradient_direction[y,x])) 
          y0 = int(x - k * np.cos(gradient_direction[y,x]))
          if x0 <= d1 - k and x0 > 0 and y0 <= d2 - k and y > 0:
            hough3D[x0,y0,k-1] = hough3D[x0,y0,k-1] + 1
  
  print("hough returns")

  return hough3D

def threshold(gradient_magnitude, T):

  threshold_image = np.zeros([gradient_magnitude.shape[0], gradient_magnitude.shape[1]], dtype=np.float64)
  for y in range(0, gradient_magnitude.shape[0]):
    for x in range(0, gradient_magnitude.shape[1]):
      if gradient_magnitude[y,x] > T:
        threshold_image[y,x] = 255
 

  return threshold_image  


hough_space = hough(gradient_magnitude, gradient_direction, 160)

hough_image = np.zeros([hough_space.shape[0], hough_space.shape[1]], dtype=np.float64)


for x in range(0, hough_space.shape[0]):
  for y in range(0, hough_space.shape[1]):
    summ = 0
    for r in range(0, hough_space.shape[2]):
      summ += hough_space[x,y,r]
      hough_image[x,y] = summ


cv2.imwrite("houghimage.jpg", hough_image);
