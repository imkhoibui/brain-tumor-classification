import ast
import os
def read_annotation(img_path):
  """
    This function reads the annotation file and returns its content
    as a list of dictionaries, each dict contains information
    regarding an image
    ----------
    Parameter:
    - img_path (str): the path of the image folder
    Return:
    - dataset_dict (dict): a list of the dataset's annotations
  """
  dataset_dict = {}
  with open(f"{img_path}/_annotations.coco.json", "r") as annotation:
    annot = annotation.read()
    annot = ast.literal_eval(annot)
    for image, annotation in list(zip(annot['images'], annot['annotations'])):
      input = {}
      file_name = os.path.join(img_path, image['file_name'])
      # Assign the image info to the input dict
      
      input['height'] = image['height']
      input['width'] = image['width']
      input['image_id'] = image['id']

      # Asserting the id is consistent between image information & its annotation
      message = f"Image id is in consistent in image file {file_name}"
      assert input['image_id'] == annotation['id'], message

      # Assign the annotation info to the input dict
      input['category_id'] = annotation['category_id']
      input['bbox'] = annotation['bbox']
      input['area'] = annotation['area']
      input['segmentation'] = annotation['segmentation']
      input['iscrowd'] = annotation['iscrowd']

      dataset_dict[file_name] = input
    return dataset_dict