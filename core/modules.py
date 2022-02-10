

def get_model_class(model_type):
  if model_type == "fcn_small":
    from .fcn import FeedForwardNet_small
    return FeedForwardNet_small

  elif model_type == "fcn_to_small":
    from .fcn import FeedForwardNet_toSmall
    return FeedForwardNet_toSmall

  elif model_type == "fcn_to_large":
    from .fcn import FeedForwardNet_toLarge
    return FeedForwardNet_toLarge

  elif model_type == "cnn":
    from .cnn import CNN
    return CNN

  elif model_type == "res1":
    from .cnn import RestCNN
    return RestCNN

  elif model_type == "cnn2":
    from .cnn import CNN2
    return CNN2

  elif model_type == "cnn_add":
    from .cnn import CNN_add
    return CNN_add

  elif model_type == "crnn":
    from .crnn import CRNN
    return CRNN
