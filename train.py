import sys
import argparse
import torch
import torch.optim as optim
import utils
import math
from dataset import TumorDataset
from model import TumorNet
from efficientnet_pytorch import EfficientNet
from torch import nn
from torch.utils.data import DataLoader
from annotation import read_annotation
from preprocessing import transform_img

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = "brain_images/train/"
    val_dir = "brain_images/valid/"
    test_dir = "brain_images/test/"

    train_annotation = read_annotation(train_dir)
    val_annotation = read_annotation(val_dir)
    test_annotation = read_annotation(test_dir) 

    train_dataset = TumorDataset(train_dir, train_annotation, transform_img)
    val_dataset = TumorDataset(val_dir, val_annotation)
    test_dataset = TumorDataset(test_dir, test_annotation)

    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    efficientnet = EfficientNet.from_pretrained('efficientnet-b2')
    for param in efficientnet.parameters():
        param.requires_grad = False
    model = TumorNet(efficientnet)

    train_epoch(model, train_loader)

    eval_epoch(model, val_loader)

def train_epoch(model, train_loader):
  num_epochs = 10
  learning_rate = 0.001
  decay_factor = 0.03
  criterion_1 = nn.CrossEntropyLoss()
  criterion_2 = nn.MSELoss()
  model.train()

  for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0.0
    total_samples = 0
    if (epoch + 1) % 5 == 0:
      learning_rate = learning_rate*(math.exp(-decay_factor*epoch))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for inputs, targets in train_loader:
      true_labels = targets['category']
      bboxs = targets['bbox']
      inputs = inputs.float()
      optimizer.zero_grad()
      pred_category, pred_bbox = model(inputs)

      _entropy_loss = criterion_1(pred_category, true_labels)
      _mse_loss = criterion_2(pred_bbox, bboxs)
      loss = (_entropy_loss + _mse_loss) / 2
      loss.backward()
      optimizer.step()
      running_loss += loss.item() * inputs.size(0)
      _, predicted = torch.max(pred_category, 1)
      correct_predictions += (predicted == true_labels).sum().item()
      total_samples += true_labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

def eval_epoch(model, val_loader):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training your model")
    parser.add_argument(
        "--pretrained_path",
        default="",
        help="",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="",
        help="",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        default="",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()