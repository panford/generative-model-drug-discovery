from dataclasses import dataclass, field

@dataclass
class Config:
  num_props:     int = field(default=3, metadata={"help":"Number of target properties"})
  latent_dim:    int = field(default=100, metadata={"help":"Dimension of latent vector"}) 
  batch_size:    int = field(default=32, metadata={"help":"Batch size"})
  embedding_dim: int = field(default=200, metadata={"help":"Embedding dimension"})
  num_chars:     int = field(default=3, metadata={"help":"Compute the number of characters or unique tokens"})
  n_units:       int = field(default=96, metadata={"help":"Number of recurrent units"})
  learning_rate: float = field(default=0.002, metadata={"help":"Learning rate for optimizer"})
  rate:          float = field(default=0.0, metadata={"help":"KL annealing rate init (Useful when you want to do KL annealing"})
  annealing_rate:float = field(default=0.0001 , metadata={"help":"KL annealing rate"})
