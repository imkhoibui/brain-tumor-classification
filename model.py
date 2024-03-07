import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class TumorNet(nn.Module):
  """
    Class for Neural Network that utilizes transfer learning of EfficientNet
  """
  def __init__(self, efficientnet):
    
    super(TumorNet, self).__init__()
    self.features = efficientnet
    self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(0.5)
    self.fc = nn.Linear(efficientnet._fc.in_features, 2)

  def forward(self, x):
    x = self.features.extract_features(x)
    x = self.global_avg_pooling(x)
    x = torch.flatten(x, 1)
    x = self.dropout(x)
    x = self.fc(x)
    return x

def __main__():
  efficientnet = EfficientNet.from_pretrained('efficientnet-b2')
  for param in efficientnet.parameters():
      param.requires_grad = False
  model = TumorNet(efficientnet)
  