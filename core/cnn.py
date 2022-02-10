import torch.nn as nn
import torch
import torch.nn.functional as F

def conv_output_size(w, kernel_size=1, stride=1, padding=0):
  return ((w-kernel_size+2*padding)/stride)+1

def init_bn(bn):
  # """Initialize a Batchnorm layer."""
  bn.bias.data.fill_(0.0)
  bn.weight.data.fill_(1.0)
  bn.running_mean.data.fill_(0.0)
  bn.running_var.data.fill_(1.0)

def init_layer(layer):
  # """Initialize a Linear or Convolutional layer."""
  nn.init.xavier_uniform_(layer.weight)

  if hasattr(layer, "bias"):
    if layer.bias is not None:
      layer.bias.data.fill_(0.0)

class CNN2(nn.Module):
  def init_weight(self):
    init_layer(self.conv1)
    init_layer(self.conv2)
    init_layer(self.fc1)
    init_layer(self.fc2)
    init_bn(self.bn1)
    init_bn(self.bn2)
    init_bn(self.bn3)

  def __init__(self, shape=(242,13), p=0):
    super(CNN2, self).__init__()
    _ks1 = 5
    _ks2 = 3
    _pad1 = 1
    # Conv output size: [(W−K+2P)/S]+1
    _x = conv_output_size(shape[0], kernel_size=_ks1, padding=_pad1)
    _x = int(_x/2) # pool
    _x = conv_output_size(_x,kernel_size=_ks2)
    _x = int(_x/2) # pool
    _y = conv_output_size(shape[1], kernel_size=_ks1, padding=_pad1)
    _y = int(_y/2) # pool
    _y = conv_output_size(_y,kernel_size=_ks2)
    _y = int(_y/2) # pool

    self.conv1 = nn.Conv2d(1, 4, kernel_size=_ks1, padding=_pad1)
    self.pool = nn.MaxPool2d(2, 2)
    self.bn1 = nn.BatchNorm2d(4)
    self.conv2 = nn.Conv2d(4, 8, kernel_size=_ks2)
    self.bn2 = nn.BatchNorm2d(8)
    self.fc1 = nn.Linear(8*_x*_y, 120)
    self.bn3 = nn.BatchNorm1d(120)
    self.fc2 = nn.Linear(120, 10)

    self.init_weight()

  def forward(self, input):
    x = self.pool(self.bn1(F.relu(self.conv1(input))))
    x = self.pool(self.bn2(F.relu(self.conv2(x))))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = self.bn3(F.relu(self.fc1(x)))
    x = self.fc2(x)
    return x

class RestCNN(nn.Module):
  def __init__(self, shape=(242,13), p=0):
    super(RestCNN, self).__init__()
    _ks1 = 5
    _ks2 = 3
    _pad1 = 1
    # Conv output size: [(W−K+2P)/S]+1
    _x = conv_output_size(shape[0], kernel_size=_ks1, padding=_pad1)
    _x = int(_x/2) # pool
    _x = conv_output_size(_x,kernel_size=_ks2)
    _x = int(_x/2) # pool
    _y = conv_output_size(shape[1], kernel_size=_ks1, padding=_pad1)
    _y = int(_y/2) # pool
    _y = conv_output_size(_y,kernel_size=_ks2)
    _y = int(_y/2) # pool

    self.conv_block = nn.Sequential(
      nn.Conv2d(1, 4, kernel_size=_ks1, padding=_pad1),
      nn.ReLU(),
      nn.BatchNorm2d(4),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(4, 8, kernel_size=_ks2),
      nn.ReLU(),
      nn.BatchNorm2d(8),
      nn.MaxPool2d(2, 2),
    )

    self.fc_block = nn.Sequential(
      nn.Linear(8*_x*_y, 120),
      nn.ReLU(),
      nn.BatchNorm1d(120),
      # nn.Linear(120, 84),
      # nn.ReLU(),
      # nn.BatchNorm1d(84),
      # nn.Linear(84, 10)
      nn.Linear(120, 10)
    )

    self.conv_skip = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=10, stride=4, padding=0),
        nn.BatchNorm2d(8),
    )

  def forward(self, input):
    x = self.conv_block(input) + self.conv_skip(input)
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = self.fc_block(x)
    return x

class CNN_add(nn.Module):
  def __init__(self, shape=(242,13), leakyRelu=False, p=0):
    super().__init__()
    ks = [5,2] # kernel size
    ss = [1,1] # stride size
    ps = [1,0] # pad size
    nm = [4,8] # channel

    cnn = nn.Sequential()

    def convRelu(i, batchNormalization=True):
        nIn = 1 if i == 0 else nm[i-1]
        nOut = nm[i]
        cnn.add_module('conv{0}'.format(i),
                    nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))

        if batchNormalization:
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))

        if leakyRelu:
            cnn.add_module('relu{0}'.format(i),
                            nn.LeakyReLU(0.2, inplace=True))
        else:
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

    convRelu(0)
    cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
    convRelu(1)
    cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))

    self.cnn = cnn

    # Conv output size: [(W−K+2P)/S]+1
    _x = conv_output_size(shape[0], kernel_size=ks[0], padding=ps[0])
    _x = int(_x/2) # pool
    _x = conv_output_size(_x,kernel_size=ks[1])
    _x = int(_x/2) # pool
    _y = conv_output_size(shape[1], kernel_size=ks[0], padding=ps[0])
    _y = int(_y/2) # pool
    _y = conv_output_size(_y,kernel_size=ks[1])
    _y = int(_y/2) # pool

    self.fcn = nn.Sequential(
      nn.Linear(8*_x*_y, 120),
      nn.ReLU(),
      nn.BatchNorm1d(120),
      nn.Linear(120, 84),
      nn.ReLU(),
      nn.BatchNorm1d(84),
      nn.Linear(84, 10)
    )

  def forward(self, input):
    x = self.cnn(input)
    x = torch.flatten(x, 1)
    x = self.fcn(x)
    return x

class CNN(nn.Module):
  def __init__(self, shape=(242,13), p=0):
    super().__init__()
    _ks1 = 5
    _ks2 = 3
    _pad1 = 1
    self.conv1 = nn.Conv2d(1, 4, kernel_size=_ks1, padding=_pad1)
    self.pool = nn.MaxPool2d(2, 2)
    self.bn1 = nn.BatchNorm2d(4)
    self.conv2 = nn.Conv2d(4, 8, kernel_size=_ks2)
    self.bn2 = nn.BatchNorm2d(8)

    # Conv output size: [(W−K+2P)/S]+1
    _x = conv_output_size(shape[0], kernel_size=_ks1, padding=_pad1)
    _x = int(_x/2) # pool
    _x = conv_output_size(_x,kernel_size=_ks2)
    _x = int(_x/2) # pool
    _y = conv_output_size(shape[1], kernel_size=_ks1, padding=_pad1)
    _y = int(_y/2) # pool
    _y = conv_output_size(_y,kernel_size=_ks2)
    _y = int(_y/2) # pool

    self.fc1 = nn.Linear(8*_x*_y, 120)
    self.bn3 = nn.BatchNorm1d(120)
    self.fc2 = nn.Linear(120, 84)
    self.bn4 = nn.BatchNorm1d(84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, input):
    x = self.pool(self.bn1(F.relu(self.conv1(input))))
    x = self.pool(self.bn2(F.relu(self.conv2(x))))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = self.bn3(F.relu(self.fc1(x)))
    x = self.bn4(F.relu(self.fc2(x)))
    x = self.fc3(x)
    return x
