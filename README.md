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


## Acknowledgement
This project is based on an approach proposed by Nature's Scientific Report - *Detection and classification of brain tumors using hybrid deep 
learning models.*

Babu Vimala, B., Srinivasan, S., Mathivanan, S.K. et al. Detection and classification of brain tumors using hybrid deep learning models. Sci Rep 13, 23029 (2023). https://doi.org/10.1038/s41598-023-50505-6
