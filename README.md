# Brain tumor classification and detection - a hybrid deep learning approach

## Overview
Cancer stands out as one of the most prominent and deadly diseases we currently face. Detecting tumors early, which can serve as potential indicators of cancer, is crucial in our fight against this formidable adversary. As AI tools continue to evolve and find applications in medicine and image diagnostics, they become valuable allies in tumor detection alongside healthcare professionals. However, it’s essential to emphasize that AI should not replace professional diagnosis. Instead, it serves as a supportive tool, assisting clinicians in their assessments.

In this project, I attempted to train a deep-learning model in brain tumor detection and classification. Results output a boundary box in each image, indicating the possible region where the tumor resides. The overall approach is to fine-tune an existing neural network architecture called EfficientNet to adapt to this specific problem.

## Dataset
The dataset is "Brain Tumor Image Dataset: Semantic Segmentation" by Roboflow exported on August 19, 2023. <br>
You can find the publishment of this dataset on [Kaggle](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation/data)

Upload the kaggle.json file on Colab or an IDE, then perform the following command to download.
```
!kaggle datasets download -d pkdarabi/brain-tumor-image-dataset-semantic-segmentation
```

The dataset already provides us with 3 separate folders - train, valid, and test; each with its own COCO annotation file. 

## Approach
### Preprocessing the images
Images were splitted into 3 folders, train, val, test. 
The initial images size were 640x640x3, indicating 3 channels (colored) images.

To work with this image type, I performed some preprocessing steps, including Gaussian blur to remove
unneccessary details, crop to borders, and then resize them to 240x240x1 for training.

### Working with annotation files
Annotation files were included in the 3 folders, under COCO (Common Objects in Context) annotation format.
I only kept information related to training, such as bbox, category_id and segmentation.

### Transfer learning the model using EfficientNet
EfficientNet is a family of convolutional neural network (CNN) architectures that makes use of many improvements
such as Compound Coefficient, Neural Architecture Search (ANS) and Compound Scaling.

In this project, I attempted to perform transfer learning on EfficientNet, where the first few layers of EfficientNet
were frozen. Fine-tuning the model was done by using a Global Average Pooling layer, followed by a Dropout layer
with 0.5 probability of dropping and ended with a fully connected layer that has 2 outputs.

The hyperparameters used in this project is of below:
- Input: 240x240x1
- Num epochs: 50
- Batch size: 32
- Learning rate (initial): 0.001
- Decay factor of learning rate: 0.3 after every 5 epochs
- Patient level: 5

## Project Results
The model used on MRI images test set provides the following result:
- Accuracy:
- Recall:
- F1:
- IoU: 
- Confusion matrix:

## Acknowledgement
This project is based on an approach proposed by Nature's Scientific Report - *Detection and classification of brain tumors using hybrid deep 
learning models.*

Babu Vimala, B., Srinivasan, S., Mathivanan, S.K. et al. Detection and classification of brain tumors using hybrid deep learning models. Sci Rep 13, 23029 (2023). https://doi.org/10.1038/s41598-023-50505-6
