import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNet_toSmall(nn.Module):
  def __init__(self, shape=(242,13), p=0.5):
      super().__init__()
      self.flatten = nn.Flatten()
      self.dense_layer = nn.Sequential(
        nn.Linear(shape[0]*shape[1], 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=p),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=p),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(p=p),
        nn.Linear(64, 10),
      )

  def forward(self, input):
    flattenned_data = self.flatten(input)
    output = self.dense_layer(flattenned_data)
    return output

class FeedForwardNet_toLarge(nn.Module):
  def __init__(self, shape=(242,13), p=0.5):
      super().__init__()
      self.flatten = nn.Flatten()
      self.dense_layer = nn.Sequential(
        nn.Linear(shape[0]*shape[1], 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(p=p),
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=p),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=p),
        nn.Linear(512, 10)
      )

  def forward(self, input):
    flattenned_data = self.flatten(input)
    output = self.dense_layer(flattenned_data)
    return output

class FeedForwardNet_small(nn.Module):
  def __init__(self, shape=(242,13), p=0.5):
      super().__init__()
      self.flatten = nn.Flatten()
      self.dense_layer = nn.Sequential(
        nn.Linear(shape[0]*shape[1], 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(p=p),
        nn.Linear(64, 10)
      )

  def forward(self, input):
    flattenned_data = self.flatten(input)
    output = self.dense_layer(flattenned_data)
    return output