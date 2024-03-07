import cv2 as cv 
import imutils
import numpy as np
from torchvision.transforms import v2

def img_border_crop(img):
  """
    This function crops the image to retain only a rectangle that fills its
    boundaries

    ----------
    Parameters:
    - img (jpg): an img passed to be cropped

    ----------
    Returns:
    - img (jpg): the img cropped
  """
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  img = cv.GaussianBlur(img, (3, 3), 0)

  img = cv.threshold(img, 45, 225, cv.THRESH_BINARY)[1]
  kernel = np.ones((3, 3), np.uint8)
  img = cv.erode(img, kernel, iterations=2)
  img = cv.dilate(img, kernel, iterations=2)

  cnts = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  c = max(cnts, key=cv.contourArea)

  left = tuple(c[c[:, :, 0].argmin()][0])
  right = tuple(c[c[:, :, 0].argmax()][0])
  top = tuple(c[c[:, :, 1].argmin()][0])
  bot = tuple(c[c[:, :, 1].argmax()][0])

  img = img[top[1]:bot[1], left[0]:right[0]]
  return img

def transform_img(img):
  """
    This function transforms the current images
    
    ----------
    Parameters:
    - img (jpg): an img passed to be cropped

    ----------
    Return:
    - img (jpg): the img cropped
    - annotation (str): the annotation file
  """
  
  transforms = v2.Compose([
    v2.RandomResizedCrop(size=(240, 240), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  img = img_border_crop(img)
  img = cv.resize(img, dsize=(240, 240), interpolation=cv.INTER_CUBIC)
  img = np.stack((img, )*3, axis = -1)
  img = img.astype(float) / 255.0
  return transforms(img)


