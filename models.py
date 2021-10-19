from tensorflow import keras 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow import Tensor
from tensorflow.keras.backend import random_normal
from tensorflow.keras.layers import (Lambda, 
                                    Input, 
                                    RepeatVector, 
                                    TimeDistributed,
                                    Embedding, LSTM, 
                                    Dense, Bidirectional, 
                                    Concatenate,
                                    LayerNormalization)      

class vae_encoder(Model):
  '''
  Encoder 

  Inputs
    num_chars (int)      : Number of unique characters in dataset
    embedding_dim (int)  : Embedding dimension
    max_seq_len (int)    : max sequence length
    latent_dim (int)     : latent dimension
    n_rnn_units (int)    : number of rnn units

  Outputs
    (Keras Model Object) : takes data inputs and returns parameters of learned latent distribution 
  '''

  def __init__(self,num_chars:  int, 
                    embedding_dim:int, 
                    max_seq_len:  int, 
                    latent_dim:   int, 
                    n_units:      int):

    super(vae_encoder, self).__init__()


    self.embedding = Embedding(num_chars, embedding_dim, input_length=max_seq_len, mask_zero=True)
    self.bilstm1 = Bidirectional(LSTM(n_units, return_sequences=True))
    self.bilstm2 = Bidirectional(LSTM(2*n_units, return_sequences=True))
    self.bilstm3 = Bidirectional(LSTM(4*n_units))

    self.layernorm1 = LayerNormalization(axis=-1)
    self.layernorm2 = LayerNormalization(axis=-1)
    self.layernorm3 = LayerNormalization(axis=-1)

    self.dense1 = Dense(latent_dim*2)
    self.mu = Dense(latent_dim)
    self.sigma = Dense(latent_dim)


  def call(self, inputs):
    x = self.embedding(inputs)
    x = self.bilstm1(x)
    x = self.layernorm1(x)
    x = self.bilstm2(x)
    x = self.layernorm2(x)
    x = self.bilstm3(x)
    x = self.layernorm3(x)
    x = self.dense1(x)
    mu = self.mu(x)
    sigma = self.sigma(x)
    return mu, sigma



class reparameterization(Layer):
  """
  Reparameterization function--> takes mean and sigma and reparameterize with samples
   drawn from a standard normal distribution with mean 0 and standard deviation 1. 

  Inputs:
   (mu, sigma): mean and standard deviation 

  Output:
    z        : latent code 

   """
  
  # def __init__(self, ):

  def call(self, mu, sigma):
    batch_ = tf.shape(mu)[0]
    dim = tf.shape(mu)[1]
    eps = random_normal((batch_, dim))
    
    return mu + tf.exp(0.5*sigma) * eps


class vae_decoder(Model):
  """
  Decoder 

  Inputs
    num_chars (int)      : Number of unique characters in dataset
    max_seq_len (int)    : max sequence length
    latent_dim (int)     : latent dimension
    n_gru_units (int)    : number of gru units

  Outputs
    (Keras Model Object) : takes latent vectors as inputs and returns a 
                           softmax distribution over character set 
  """
  def __init__(self,num_chars:  int, 
                    max_seq_len:int, 
                    latent_dim: int, 
                    n_units:    int):
    super(vae_decoder, self).__init__()

    self.repeatvect = RepeatVector(max_seq_len)
    self.lstm1 = LSTM(n_units, return_sequences=True)
    self.layernorm1 = LayerNormalization(axis=-1)
    self.lstm2 = LSTM(2*n_units, return_sequences=True)
    self.layernorm2 = LayerNormalization(axis = -1)
    self.lstm3 = LSTM(4*n_units, return_sequences=True)
    self.layernorm3 = LayerNormalization(axis=-1)
    self.final_dense = Dense(num_chars, activation='softmax')

  def call(self, inputs):
    x = self.repeatvect(inputs)
    x = self.lstm1(x)
    x = self.layernorm1(x)
    x = self.lstm2(x)
    x = self.layernorm2(x)
    x = self.lstm3(x)
    x = self.layernorm3(x)
    x = self.final_dense(x)
    return x




class kl_divergence_loss():
  """ 
  Computes the Kullback-Leibler Divergence (KLD) loss
  Inputs
    inputs:  batch from the dataset
    outputs: Output from the sample_z function/ layer
    mu:      mean
    sigma:   standard deviation

  Outputs:
    KL Divergence loss
  # """
  def __init__(self, rate=0.5):
    self.rate = rate

  def __call__(self, mu, sigma):
    kl_loss = 1. + sigma - tf.square(mu) - tf.math.exp(sigma)
    kl_loss = -self.rate * tf.reduce_mean(kl_loss)
    return kl_loss 

class vae_model(Model):
  """
  Builds a complete VAE model

  Inputs
    encoder     : the encoder model
    decoder     : the decoder model
    max_seq_len : length of sequence batch

  Output:
    the complete VAE model
  """
  def __init__(self,
              num_chars:  int, 
              embedding_dim:int, 
              max_seq_len:  int, 
              latent_dim:   int, 
              n_units:      int,
              kl_div_loss_rate:float):

    super(vae_model, self).__init__()
    self.encoder = vae_encoder(num_chars, 
                              embedding_dim, 
                              max_seq_len, 
                              latent_dim, 
                              n_units)

    self.decoder = vae_decoder(num_chars, max_seq_len, latent_dim, n_units)
    self.reparameterization = reparameterization()
    self.kl_loss_fn = kl_divergence_loss(kl_div_loss_rate)
    self.kl_loss = 0
    self.mu = None
    self.sigma = None

  def compute_kl_loss(self, mu, sigma):
    self.kl_loss = self.kl_loss_fn(mu, sigma)
  
  def call(self, inputs):
    self.mu, self.sigma = self.encoder(inputs)
    
    z = self.reparameterization(self.mu, self.sigma)
   
    reconstructed = self.decoder(z)

    self.compute_kl_loss(self.mu, self.sigma)
 
    return reconstructed

# Initialize vae model
def init_vae_models(num_chars, embedding_dim, max_seq_len, latent_dim, n_units, kl_div_loss_rate):
  """ Model initializations here """
  vae = vae_model(num_chars, embedding_dim, max_seq_len, latent_dim, n_units, kl_div_loss_rate)  
  return vae




