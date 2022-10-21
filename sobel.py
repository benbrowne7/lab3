import argparse
import cv2
import numpy as np
import math


parser = argparse.ArgumentParser(description='Sobel')
parser.add_argument('-name', '-n', type=str, default='coins1.png')

args = parser.parse_args()
iname = args.name
g = cv2.imread(iname, 1)
image = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY );

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

        
          

def sobel(image):
  x_derivative, x_new = xderivative(image)
  y_derivative, y_new = yderivative(image)
  gradient_magnitude = gradientmag(x_new,y_new)
  gradient_direction = directionp(x_new,y_new)

  cv2.imwrite("xderi.jpg", x_derivative);
  cv2.imwrite("yderi.jpg", y_derivative);
  cv2.imwrite("gradientmag.jpg", gradient_magnitude);
  cv2.imwrite("gradientdirection.jpg", gradient_direction);

  print("sobel returns")

  return gradient_magnitude, gradient_direction

def threshold(gradient_magnitude, T):
  threshold_image = np.zeros([gradient_magnitude.shape[0], gradient_magnitude.shape[1]], dtype=np.float64)
  for y in range(0, gradient_magnitude.shape[0]):
    for x in range(0, gradient_magnitude.shape[1]):
      if gradient_magnitude[y,x] > T:
        threshold_image[y,x] = 255
 

  return threshold_image


def padinput(image, kernel):
  
  kernelRadiusX = round(( kernel.shape[0] - 1 ) / 2)
  kernelRadiusY = round(( kernel.shape[1] - 1 ) / 2)
	
  paddedimage = cv2.copyMakeBorder(image, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, 
		cv2.BORDER_REPLICATE)
  
  return paddedimage

def directionp(x_derivative, y_derivative):
  directionp = np.zeros([x_derivative.shape[0], x_derivative.shape[1]], dtype=np.float64)
  for y in range(0, x_derivative.shape[0]):
    for x in range(0, x_derivative.shape[1]):
      if x_derivative[y,x] == 0.0000000000000000:
        pixelatan = 90
      else:
        pixelatan = math.degrees(math.atan(float(y_derivative[y,x])/float(x_derivative[y,x])))
        
      directionp[y,x] = pixelatan
  return directionp

def gradientmag(x_derivative, y_derivative):
  gradientmag = np.zeros([x_derivative.shape[0], x_derivative.shape[1]], dtype=np.float64)
  for y in range(0, x_derivative.shape[0]):
    for x in range(0, x_derivative.shape[1]):
      pixelmag = np.sqrt((x_derivative[y,x]**2) + (y_derivative[y,x]**2))
      gradientmag[y,x] = pixelmag
      
  
  return gradientmag

def xderivative(image):

  x_derivative = np.zeros([image.shape[0], image.shape[1]], dtype=np.float64)
  xkernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  paddedinput = padinput(image, xkernel)
  for y in range(1, paddedinput.shape[0]-1):	
    for x in range(1, paddedinput.shape[1]-1):
      
      a1 = paddedinput[y - (-1), x - (-1)] * xkernel[0,0]  # h(-1,-1)
      a2 = paddedinput[y - (-1), x - (0)] * xkernel[0,1]  # h(-1,0)
      a3 = paddedinput[y - (-1), x - (1)] * xkernel[0,2] # h(-1,1)
      a4 = paddedinput[y - (0), x - (-1)] * xkernel[1,0] # h(0,-1)
      a5 = paddedinput[y - (0), x - (0)] * xkernel[1,1] # h(0,0)
      a6 = paddedinput[y - (0), x - (1)] * xkernel[1,2] # h(0,1)
      a7 = paddedinput[y - (1), x - (-1)] * xkernel[2,0] # h(1,-1)
      a8 = paddedinput[y - (1), x - (0)] * xkernel[2,1] # h(1,0)
      a9 = paddedinput[y - (1), x - (1)] * xkernel[2,2] # h(1,1)
      summation = a1+a2+a3+a4+a5+a6+a7+a8+a9
      x_derivative[y-1,x-1] = float(summation)

  x_new = cv2.filter2D(image,ddepth=-1,kernel=xkernel)



  return x_derivative, x_new
  
def yderivative(image):
  
  y_derivative = np.zeros([image.shape[0], image.shape[1]], dtype=np.float64)
  ykernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
  paddedinput = padinput(image, ykernel)
  for y in range(1, paddedinput.shape[0]-1):	
    for x in range(1, paddedinput.shape[1]-1):
      
      a1 = paddedinput[y - (-1), x - (-1)] * ykernel[0,0]  # h(-1,-1)
      a2 = paddedinput[y - (-1), x - (0)] * ykernel[0,1]  # h(-1,0)
      a3 = paddedinput[y - (-1), x - (1)] * ykernel[0,2] # h(-1,1)
      a4 = paddedinput[y - (0), x - (-1)] * ykernel[1,0] # h(0,-1)
      a5 = paddedinput[y - (0), x - (0)] * ykernel[1,1] # h(0,0)
      a6 = paddedinput[y - (0), x - (1)] * ykernel[1,2] # h(0,1)
      a7 = paddedinput[y - (1), x - (-1)] * ykernel[2,0] # h(1,-1)
      a8 = paddedinput[y - (1), x - (0)] * ykernel[2,1] # h(1,0)
      a9 = paddedinput[y - (1), x - (1)] * ykernel[2,2] # h(1,1)
      summation = a1+a2+a3+a4+a5+a6+a7+a8+a9
      y_derivative[y-1,x-1] = float(summation)

  y_new = cv2.filter2D(image, ddepth=-1, kernel=ykernel)

  return y_derivative, y_new


gradient_magnitude, gradient_direction = sobel(image)


