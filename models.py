from tensorflow import keras 
from tensorflow.keras import Model
from tensorflow import Tensor
from tensorflow.keras.layers import (Lambda, 
                                    Input, 
                                    RepeatVector, 
                                    TimeDistributed,
                                    Embedding, LSTM, 
                                    Dense, Bidirectional, 
                                    Concatenate)      

def encoder_model(num_chars:  int, 
                embedding_dim:int, 
                max_seq_len:  int, 
                latent_dim:   int, 
                n_units:      int):

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

  inputs = Input(shape = (max_seq_len, ))
  embedding = Embedding(num_chars, embedding_dim, input_length=max_seq_len, mask_zero=True)(inputs)
  x = Bidirectional(LSTM(n_units))(embedding)
  x = Dense(latent_dim*2)(x)
  mu = Dense(latent_dim)(x)
  sigma = Dense(latent_dim)(x)
  model = tf.keras.Model(inputs, outputs = [mu, sigma])
  return model

# plot_model(encoder_model(num_chars, embedding_dim, max_seq_len, latent_dim, n_units), dpi=70)


def reparameterization(inputs:Tensor):

  
  """
  Reparameterization function--> takes mean and sigma and reparameterize with samples
   drawn from a standard normal distribution with mean 0 and standard deviation 1. 

  Inputs:
   (mu, sigma): mean and standard deviation 

  Output:
    z        : latent code 

   """
  mu, sigma = inputs
  batch_ = tf.shape(mu)[0]
  dim = tf.shape(mu)[1]
  eps = random_normal((batch_, dim))
  
  return mu + tf.exp(0.5*sigma) * eps

def decoder_model(num_chars:  int, 
                  max_seq_len:int, 
                  latent_dim: int, 
                  n_units:    int):

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


  inputs = Input(shape = (latent_dim))
  x = RepeatVector(max_seq_len)(inputs)
  lstm_out = LSTM(n_units, return_sequences=True)(x)
  output = Dense(num_chars, activation='softmax')(lstm_out)

  model = Model(inputs, output)
  return model



def kl_divergence_loss(inputs:Tensor, 
                      outputs:Tensor, 
                      mu:float, 
                      sigma:float):
  
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

  rate = 0.5
  kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
  kl_loss = -rate * tf.reduce_mean(kl_loss)

  return kl_loss 

def vae_model(encoder:Model, decoder:Model, max_seq_len:int):

  
  """
  Biulds a complete VAE model

  Inputs
    encoder     : the encoder model
    decoder     : the decoder model
    max_seq_len : length of sequence batch

  Output:
    the complete VAE model
  """

  # set the inputs
  input_x = tf.keras.layers.Input(shape=(max_seq_len, ))

  # get mu, sigma, and z from the encoder output
  mu, sigma = encoder(input_x)
  
  z = Lambda(reparameterization)(([mu, sigma]))
  # get reconstructed output from the decoder
  reconstructed = decoder(z)

  # define the inputs and outputs of the VAE
  model = tf.keras.Model(inputs=input_x, outputs=reconstructed)

  # add the KL loss
  loss = kl_divergence_loss(input_x, z, mu, sigma)
  model.add_loss(loss)

  return model


