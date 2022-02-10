import torch
from torch.optim import optimizer
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch import nn
import yaml
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import argparse

import preproc
from core.modules import get_model_class

# Fix seed
seed = 19
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print('seed:', torch.initial_seed())

class MyDataSet(Dataset):
  def __init__(self, **data):
      super().__init__()
      self.data = torch.tensor(data["mfcc"])
      self.label = torch.LongTensor(data["labels"])

  def __getitem__(self,index):
    return self.data[index], self.label[index]

  def __len__(self):
    return len(self.data)

def validation(model, val_data_loader, loss_fn, device):
  # switch mode
  model.eval()

  for input, target in val_data_loader:
    input, target = input.to(device), target.to(device)
    # Calculate loss
    input = input[None,...]
    input = input.reshape(-1, 1, data_shape[0], data_shape[1]) # (-1, 1, 242, 13)
    prediction = model(input)
    loss = loss_fn(prediction, target)

  # switch mode
  model.train()

  return loss

def train_one_epoch(model, data_loader, val_data_loader, loss_fn, optimiser, device, data_shape):
  model.train()
  for input, target in data_loader:
    input, target = input.to(device), target.to(device)
    # Calculate loss
    input = input[None,...]
    input = input.reshape(-1, 1, data_shape[0], data_shape[1]) # (-1, 1, 242, 13)
    prediction = model(input)
    loss = loss_fn(prediction, target)
    # backpropogate loss and update weights
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
    loss_val = validation(model, val_data_loader, loss_fn, device)


  print('loss (train, val) = ({},{})'.format(loss.item(), loss_val.item()))

  return loss, loss_val

def train(model, train_data_loader, val_data_loader, loss_fn, optimiser, device,
        init_epoch, epochs, data_shape, writer=None):
  loss_train_list = []
  loss_val_list = []
  acc_train_list = []
  acc_val_list = []
  best_acc = 0.0
  iter = 0
  for i in range(init_epoch, epochs+1):
    print('epoch', i)
    loss_train, loss_val = train_one_epoch(model, train_data_loader, val_data_loader, 
                                          loss_fn, optimiser, device, data_shape)
    loss_train_list.append(loss_train.item())
    loss_val_list.append(loss_val.item())
    if writer:
      writer.add_scalars("Loss", {
        'train loss': loss_train.item(),
        'val loss': loss_val.item(),
        }, i-1)
    # store model state
    state = model.state_dict()
    torch.save(state, 'checkpoints/model_{}.pth'.format(i))
    # store optimizer state
    opt_state = optimiser.state_dict()
    torch.save(opt_state, 'checkpoints/optimizer_{}.pth'.format(i))

    train_acc = eval(model, device, train_data_loader, data_shape)
    # validation
    acc = eval(model, device, val_data_loader, data_shape)
    acc_train_list.append(train_acc)
    acc_val_list.append(acc)
    if writer:
      writer.add_scalars("Acc", {
        'train acc': train_acc,
        'val acc': acc,
      }, i-1)

    iter += 1
    # store best model
    if acc >= best_acc:
      best_acc = acc
      torch.save(state, 'checkpoints/model_best.pth')
      global_state = {'global_step': iter, 'acc': acc}
      torch.save(global_state, 'checkpoints/global_info')
      print('update best acc {} in iteration {}'.format(acc, iter))
 
  plt.subplot(211)
  plt.plot(loss_train_list, label='Training loss')
  plt.plot(loss_val_list, label='Validation loss')
  plt.legend()
  plt.grid()
  plt.subplot(212)
  plt.plot(acc_train_list, label='Training acc')
  plt.plot(acc_val_list, label='Validation acc')
  plt.legend()
  plt.grid()

@torch.no_grad()
def eval(model, device, data_loader, data_shape):
  model = model.eval()
  num_all= torch.tensor(0.)
  num_tp = torch.tensor(0.)

  for batch_idx, (_mfcc, _label) in enumerate(data_loader):
    _mfcc, _label = _mfcc.to(device), _label.to(device, dtype=torch.int64)
    # inference
    _mfcc = _mfcc[None,...]
    _mfcc = _mfcc.reshape(-1, 1, data_shape[0], data_shape[1])
    predict_label = model(_mfcc)
    predict_label = nn.Softmax(1)(predict_label)
    predict = torch.argmax(predict_label, 1)
        
    # Count true positive
    num_all = num_all + predict.shape[0]
    num_tp = num_tp + (_label == predict).sum()
    predict_label = predict_label.cpu().numpy()

  acc = num_tp.item() / num_all.item()
  print('Accuracy: {}  ({}/{})'.format(num_tp.item() / num_all.item(),
                                num_tp.item(),
                                num_all.item()))
  return acc

if __name__ == "__main__":
  # tensorboad writer
  writer = SummaryWriter('runs/genre_classification')

  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True, help='assign the model')
  parser.add_argument('--db', default='big', help='database (small or big)')
  parser.add_argument('--init_epoch', type=int, default=1, help='initial epoch')
  parser.add_argument('--n_epoch', type=int, default=30, help='number of epoch')
  parser.add_argument('--p', type=float, default=0, help='dropout')
  opt = parser.parse_args()
  print(opt)

  # create folder
  checkpoint_dir = "checkpoints/"
  os.makedirs(checkpoint_dir, exist_ok=True)

  # load parameters
  with open('configs/baseline.yaml', 'r') as f:
    config=yaml.safe_load(f)

  # create a train data set
  # data_json, sr = preproc.save_mfcc_json(n_mfcc=config['n_mfcc'], num_segments=config['n_segments'])

  # load preprocessed data
  if opt.db == 'big':
    jsonpath = 'predata.json'
  else:
    jsonpath = 'predata_small.json'
  data_json = preproc.load_mfcc_jason(json_path=jsonpath)

  sr = data_json['sr']
  my_data = MyDataSet(**data_json)
  data_shape = preproc.get_mfcc_size(sr=sr, n_mfcc=config['n_mfcc'], num_segments=config['n_segments'])
  print('data_shape: ', data_shape)
  
  # define dataset ratio
  data_ratio = (0.8, 0.1, 0.1) # train, val, test
  data_size = len(my_data)
  train_size = int(data_ratio[0] * data_size)
  validate_size = int(data_ratio[1] * data_size)
  test_size = data_size - train_size - validate_size
  print("train size", train_size, ", validate size", validate_size, ", test size", test_size)
  # split the dataset
  train_set, val_set, test_set = torch.utils.data.random_split(
    my_data, (train_size, validate_size, test_size)
  )
  
  # create a dataloader
  train_data_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
  val_data_loader = DataLoader(val_set, batch_size=config['batch_size']//2)
  test_data_loader = DataLoader(test_set, batch_size=20, shuffle=False)

  # build a model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('device:', device)
  
  def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        #m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)
        m.weight.data.fill_(1.0)
        m.running_mean.data.fill_(0.0)
        m.running_var.data.fill_(1.0)

  # fcn: fcn_small, fcn_to_small, fcn_to_large
  # cnn: cnn, cnn2 (w/ layer initialized), res1, cnn_add
  Net = get_model_class(opt.model)
  net = Net(data_shape, p=opt.p).to(device)
  net.apply(weights_init)

  # torchsummary cannot show rnn-like network
  if opt.model != 'crnn':
    from torchsummary import summary
    summary(net, (1, data_shape[0], data_shape[1]))

  # init loss funciton and optimiser
  loss_fn = nn.CrossEntropyLoss()
  # optimiser = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=0.001)
  optimiser = torch.optim.AdamW(net.parameters(), lr=config['lr'])
  # check if the previous epoch exists
  init_epoch = opt.init_epoch
  if init_epoch > 1:
    prev_epoch_path = 'checkpoints/model_{}.pth'.format(init_epoch-1)
    print('prev_epoch_path:',prev_epoch_path)
    assert os.path.isfile(prev_epoch_path) == True
    state = torch.load(prev_epoch_path)
    net.load_state_dict(state)
    # load optimizer
    prev_epoch_opt_path = 'checkpoints/optimizer_{}.pth'.format(init_epoch-1)
    assert os.path.isfile(prev_epoch_opt_path) == True
    opt_state = torch.load(prev_epoch_opt_path)
    optimiser.load_state_dict(opt_state)
  
  n_epoch = opt.n_epoch
  # train model
  start = time.time()
  train(net, train_data_loader, val_data_loader, loss_fn, optimiser, device,
        init_epoch, n_epoch, data_shape, writer)
  end = time.time()
  print('process time (sec):', end-start)

  print("epoch: {}, train batch size: {}, validate batch size: {}".format(
    n_epoch, config['batch_size'], config['batch_size']//2
  ))
  # evaluation
  print("===================== test evaluation =====================")
  print("----- evaluate last model -----")
  print("Train set:")
  eval(net, device, train_data_loader, data_shape)
  print("Validate set:")
  eval(net, device, val_data_loader, data_shape)
  print("Test set:")
  eval(net, device, test_data_loader, data_shape)
  # load best validation model
  print("----- evaluate best validation model -----")
  state = torch.load("checkpoints/model_best.pth")
  net.load_state_dict(state)
  print("Train set:")
  eval(net, device, train_data_loader, data_shape)
  print("Validate set:")
  eval(net, device, val_data_loader, data_shape)
  print("Test set:")
  eval(net, device, test_data_loader, data_shape)
  print("===================== end =====================")

  plt.show()