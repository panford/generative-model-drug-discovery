from dataclasses import dataclass, field, asdict, fields

@dataclass
class Config:

  epochs:  int = field(default=20, 
                        metadata={'help':"number of training epochs"})
  num_props:     int = field(default=3, 
                       metadata={"help":"Number of target properties"})
  latent_dim:    int = field(default=100, 
                       metadata={"help":"Dimension of latent vector"}) 
  batch_size:    int = field(default=32, 
                        metadata={"help":"Batch size"})
  embedding_dim: int = field(default=200, 
                        metadata={"help":"Embedding dimension"})
  num_chars:     int = field(default=3, 
                        metadata={"help":"Compute the number of characters or unique tokens"})
  n_units:       int = field(default=96, 
                        metadata={"help":"Number of recurrent units"})
  learning_rate: float = field(default=0.002, 
                        metadata={"help":"Learning rate for optimizer"})
  rate:          float = field(default=0.0, 
                        metadata={"help":"KL annealing rate init (Useful when you want to do KL annealing"})
  annealing_rate:float = field(default=0.0001, 
                        metadata={"help":"KL annealing rate"})

  def __post_init__(self):
    self.__class__.field_types = {f.name:f.type for f in fields(self.__class__)}

  @classmethod
  def init_from_file(cls, config_file):
    kwargs = {}
    with open(config_file, 'r') as f:
      lines = f.readlines()

    for line in lines:
      key, val = line.split()
      kwargs[key] = cls.field_types[key](val)
    return cls(**kwargs)

  @classmethod
  def update_config(cls, kwargs):
    for key, val in kwargs.items():
      if key in cls.__dataclass_fields__:
        cls.__dataclass_fields__[key] = val

# config = Config()
# # config = config.init_from_file('./config_file.txt')
# # config.update_config({'num_chars':5})
# # print(config)