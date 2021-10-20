from tensorflow import keras 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow import Tensor
from tensorflow.keras.backend import random_normal
from collections import defaultdict
from tensorflow.keras.layers import (Lambda, 
                                    Input, 
                                    RepeatVector, 
                                    TimeDistributed,
                                    Embedding, LSTM, 
                                    Dense, Bidirectional, 
                                    Concatenate,
                                    LayerNormalization, 
                                    LeakyReLU)
    
class EncoderBase(Model):
  def __init__(self,
                  num_chars:  int, 
                  embedding_dim:int, 
                  max_seq_len:  int, 
                  latent_dim:   int, 
                  n_units:      int):
    super(EncoderBase, self).__init__()

    # self.num_chars = num_chars
    # self.embedding_dim = embedding_dim
    # self.max_seq_len = max_seq_len
    # self.latent_dim = latent_dim
    # self.n_units = n_units


    self.embedding = Embedding(num_chars, embedding_dim, input_length=max_seq_len, mask_zero=True)
    self.bilstm1 = Bidirectional(LSTM(n_units, return_sequences=True))
    self.bilstm2 = Bidirectional(LSTM(2*n_units, return_sequences=True))
    self.bilstm3 = Bidirectional(LSTM(4*n_units))

    self.layernorm1 = LayerNormalization()
    self.layernorm2 = LayerNormalization()
    self.layernorm3 = LayerNormalization()

    self.dense = Dense(latent_dim*2)

  def compute_base_features(self, inputs):
    x = self.embedding(inputs)
    x = self.bilstm1(x)
    x = self.layernorm1(x)
    x = self.bilstm2(x)
    x = self.layernorm2(x)
    x = self.bilstm3(x)
    x = self.layernorm3(x)
    out = self.dense(x)
    return out



class DecoderBase(Model):
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
                    n_units:    int):
    super(DecoderBase, self).__init__()

    self.num_chars = num_chars
    self.max_seq_len = max_seq_len
    self.n_units = n_units


    self.repeatvect =  RepeatVector(max_seq_len)
    self.lstm1 =       LSTM(n_units, return_sequences=True)
    self.layernorm1 =  LayerNormalization()
    self.lstm2 =       LSTM(2*n_units, return_sequences=True)
    self.layernorm2 =  LayerNormalization()
    self.lstm3 =       LSTM(4*n_units, return_sequences=True)
    self.layernorm3 =  LayerNormalization()
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



class Vanilla_Encoder(EncoderBase):
  def __init__(self,num_chars:  int, 
                    embedding_dim:int,
                    max_seq_len:int, 
                    latent_dim:   int, 
                    n_units:      int):


    super(Vanilla_Encoder, self).__init__(num_chars, 
                                          embedding_dim, 
                                          max_seq_len, 
                                          latent_dim, 
                                          n_units)
    self.final_layer = Dense(latent_dim)

  def call(self, inputs):
    x = self.compute_base_features(inputs)
    return self.final_layer(x)


class VAE_Encoder(EncoderBase):
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
                    max_seq_len:int, 
                    latent_dim:   int, 
                    n_units:      int):

    super(VAE_Encoder, self).__init__(num_chars, embedding_dim, max_seq_len, latent_dim, n_units)

    self.mu = Dense(latent_dim)
    self.sigma = Dense(latent_dim)


  def call(self, inputs):
    x = self.compute_base_features(inputs)
    mu = self.mu(x)
    sigma = self.sigma(x)
    return mu, sigma

class Decoder(DecoderBase):
  def __init__(self, num_chars:  int, 
                    max_seq_len:int, 
                    n_units: int):
    super(Decoder, self).__init__(num_chars, max_seq_len, n_units)


class Vanilla_Autoencoder(Model):
  """
  Builds a complete Vanilla Autoencoder model

  Inputs
    encoder     : the encoder model
    decoder     : the decoder model
    max_seq_len : length of sequence batch

  Output:
    the complete AE model
  """
  def __init__(self, num_chars:  int, 
              embedding_dim:int, 
              max_seq_len:  int, 
              latent_dim:   int, 
              n_units:      int):

    super(Vanilla_Autoencoder, self).__init__()

    self.input_args = defaultdict(num_chars = num_chars,
                      embedding_dim = embedding_dim,
                      max_seq_len = max_seq_len,
                      latent_dim = latent_dim,
                      n_units = n_units)

    self.encoder = Vanilla_Encoder(num_chars, 
                              embedding_dim, 
                              max_seq_len, 
                              latent_dim, 
                              n_units)

    self.decoder = DecoderBase(num_chars, max_seq_len, n_units)
  
  def call(self, inputs):
    z = self.encoder(inputs)
    reconstructed = self.decoder(z)
    return z, reconstructed


class reparameterization(Layer):
  """
  Reparameterization function--> takes mean and sigma and reparameterize with samples
   drawn from a standard normal distribution with mean 0 and standard deviation 1. 

  Inputs:
   (mu, sigma): mean and standard deviation 

  Output:
    z        : latent code 

   """
  def call(self, mu, sigma):
    batch_ = tf.shape(mu)[0]
    dim = tf.shape(mu)[1]
    eps = random_normal((batch_, dim)) 
    return mu + tf.exp(0.5*sigma) * eps



class VAE_Decoder(DecoderBase):
  def __init__(self, num_chars:  int, 
                    max_seq_len:int, 
                    n_units: int):
    super(VAE_Decoder, self).__init__(num_chars, max_seq_len, n_units)


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

class VAE_model(Model):
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

    super(VAE_model, self).__init__()
    self.encoder = VAE_Encoder(num_chars, 
                              embedding_dim, 
                              max_seq_len, 
                              latent_dim, 
                              n_units)

    self.decoder = VAE_Decoder(num_chars, max_seq_len, n_units)
    self.reparameterization = reparameterization()
    self.kl_loss_fn = kl_divergence_loss(kl_div_loss_rate)
    # self.mu = None
    # self.sigma = None
    self.__kl_loss = 0

  def compute_kl_loss(self, mu, sigma):
    self.__kl_loss = self.kl_loss_fn(mu, sigma)
  
  def call(self, inputs):
    mu, sigma = self.encoder(inputs)

    z = self.reparameterization(mu, sigma)

    reconstructed = self.decoder(z)
    self.compute_kl_loss(mu, sigma)
    return z, reconstructed

  @property
  def kl_loss(self):
    return self.__kl_loss

# Initialize vae model
def init_vae_model(num_chars, embedding_dim, 
                  max_seq_len, latent_dim, n_units, 
                  kl_div_loss_rate, **kwargs):

  """ Model initializations here """
  vae = VAE_model(num_chars, embedding_dim, 
                  max_seq_len, latent_dim, n_units, 
                  kl_div_loss_rate)  
  return vae

    

class Discriminator_AE(Model):

  """
  Define discriminator
  Inputs
    latent_dim: latent dimension

  Outputs
    discriminator model 
  """

  def __init__(self, hidden_units):
    super(Discriminator_AE, self).__init__()
    self.hidden_units = hidden_units

    self.dense1 = Dense(hidden_units)
    self.dense2 = Dense(2*hidden_units)
    self.dense3 = Dense(4*hidden_units)
    self.lrelu = LeakyReLU(alpha=0.2)
    self.classifier = Dense(1, activation='sigmoid')

  def call(self, inputs):
    x = self.lrelu(self.dense1(inputs))
    x = self.lrelu(self.dense2(x))
    x = self.lrelu(self.dense3(x))
    return self.classifier(x)


class Adversarial_Autoencoder(Model):
  def __init__(self, num_chars, 
                    embedding_dim, 
                    max_seq_len, 
                    latent_dim, 
                    n_units, 
                    kl_div_loss_rate, 
                    do_variational=False):

    super(Adversarial_Autoencoder, self).__init__()
    self.do_variational = do_variational

    if do_variational:
      self.gen = VAE_model(num_chars, embedding_dim, max_seq_len, latent_dim, n_units, kl_div_loss_rate)
    else: 
      self.gen = Vanilla_Autoencoder(num_chars, embedding_dim, max_seq_len, latent_dim, n_units)

    self.disc = Discriminator_AE(32)



def init_aae_model(num_chars, embedding_dim, max_seq_len, latent_dim, 
                      n_units, do_variational, kl_div_loss_rate, **kwargs):
  if do_variational and kl_div_loss_rate is None:
    AssertionError ("Kl div loss rate should be provided to do variational")
  """Initialize models """

  adversarial_autoencoder = Adversarial_Autoencoder(num_chars, embedding_dim, 
                            max_seq_len, latent_dim, n_units, do_variational, 
                            kl_div_loss_rate)

  return adversarial_autoencoder

