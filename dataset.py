import torch
import os
import cv2 as cv
import numpy as np
from annotation import read_annotation
from preprocessing import img_border_crop, transform_img
from torch.utils.data import Dataset

class TumorDataset(Dataset):
  """
    This class is designed to load and process images along with their
    corresponding bounding box annotations and labels 
  """

  def __init__(self, img_dir, annotation, transforms=None):
    """
      A constructor for the TumorDataset class

      ----------
      Parameters:
      - img_dir (string): a string path to the directory of images
      - annotation (dict): a dictionary of image annotations
      - transforms (callable, optional): optional transformations applied to images      
    """
    super(Dataset, self).__init__()

    self.img_dir = img_dir
    self.annotation = annotation
    self.transform = transforms

  def __len__(self):
    """
      A function to return the length of the dataset

      ----------
      Returns:
      (int) the number of images in the dataset
    """
    return len(self.annotation)

  def __getitem__(self, index):
    """
      A function that fetches the item at specified index

      ----------
      Parameters:
      index (int): the index of the item

      ----------
      Returns:
      transformed_image, label (tuple): a tuple containing the image after
      transformation and its target
    """
    img_key = self.img_dir[index]
    annotation = self.annotation[img_key]
    image, target = self._load_image(annotation)

    if self.transform:
      image, target = self.transform(image, target)  

    return image, target
  
  def _load_image(self, annotation):
    """
      A function that loads an image and its target features
      
      ----------
      Parameters:
      - annotation (dict): a dictionary of image features

      ----------
      Returns:
      image, targets (tuple): a tuple containing the image and its targets

    """

    filepath = self.img_dir()
    image = cv.imread(filepath)

    image = torch.from_numpy(image).permute(2, 0, 1)
    targets = {
        'bbox': torch.as_tensor(annotation['bbox'], dtype=torch.float32),
        'category' : None,
    }
    label = annotation['category_id']
    if annotation['category_id'] == 1:
      label = 0
    elif annotation['category_id'] == 2:
      label = 1
    targets['category'] = torch.as_tensor(label, dtype=torch.int64)
    
    return image, targets