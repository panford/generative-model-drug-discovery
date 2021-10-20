from dataclasses import dataclass, field, asdict, fields
from posixpath import join
import pickle
import os
from copy import copy

@dataclass
class BaseConfig:
  def __post_init__(self):
    self.__class__.field_types = {f.name:f.type for f in fields(self.__class__)}

  def init_from_file(self, config_file):
    kwargs = {}
    with open(config_file, 'r') as f:
      lines = f.readlines()

    for line in lines:
      key, val = line.split()
      if hasattr(self, key):
        setattr(self, key, val)

  def update(self, kwargs):
    if isinstance(kwargs, dict):
      config_dict = copy(kwargs)
    elif isinstance(vars(kwargs), dict):
      config_dict = copy(vars(kwargs))
    else:
      raise TypeError ("No dictionary format found for input!")
          
    for key, value in config_dict.items():
      if hasattr(self, key):
        setattr(self, key, value)


  def save(self, config_path):
    config_file = os.path.join(config_path, (self.__class__.__name__+'.pkl'))
    with open(config_file, 'wb') as config:
      pickle.dump(asdict(self), config)

  def load(self, config_path):
    config_file = os.path.join(config_path, (self.__class__.__name__+'.pkl'))
    with open(config_file, 'rb') as config:
      config_dict = pickle.load(config)
    self.update(config_dict)


@dataclass
class ModelConfig(BaseConfig):

  latent_dim:    int = field(default=100, 
                       metadata={"help":"Dimension of latent vector"}) 
  embedding_dim: int = field(default=200, 
                        metadata={"help":"Embedding dimension"})
  num_chars:     int = field(default=3, 
                        metadata={"help":"Compute the number of characters or unique tokens"})
  n_units:       int = field(default=96, 
                        metadata={"help":"Number of recurrent units"})
  kl_div_loss_rate:          float = field(default=0.0, 
                        metadata={"help":"KL annealing rate init (Useful when you want to do KL annealing"})
  max_seq_len: float = field(default=30, 
                        metadata={"help":"max sequence length"})


@dataclass
class VaeConfig(ModelConfig):
  pass

@dataclass
class AdversarialAEConfig(ModelConfig):
  do_variational :bool = field(default=False)


@dataclass
class TrainingConfig(BaseConfig):
  epochs:  int = field(default=20, 
                      metadata={'help':"number of training epochs"})
  learning_rate: float = field(default=0.002, 
                      metadata={"help":"Learning rate for optimizer"})
  batch_size:    int = field(default=32, 
                      metadata={"help":"Batch size"})


@dataclass
class PathsConfig(BaseConfig):
  checkpoint_dir : str = field(default='./checkpoints')
  data_path      : str = field(default="./data/smiles.txt")
  save_model_dir : str = field(default = './saved_models')
  config_path    : str = field(default='./configs')
  checkpoint_file: str = field(default='model.chkpt')


# config = VaeConfig()


# config.print_repr("./config_path")
# # config = asdict(config
# # )
# print(config)

# def print_args(*args, **kwargs):
#   for i,j in kwargs.items():
#     print(i, j)
# print_args(**vars(config))
# config.init_from_file('./config_file.txt')
# config.update_config({'num_chars':5})
# print(config)