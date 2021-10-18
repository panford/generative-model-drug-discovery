import tensorflow.keras as keras
import tensorflow as tf
import argparse
from prepare_data import smiles_data
from tensorflow.keras.preprocessing.text import Tokenizer
from models import init_vae_models
from config import Config 
from tensorflow.keras import losses
from tensorflow.keras import optimizers

tokenizer = Tokenizer(filters=None,
                      lower=False,
                      char_level=True)


parser = argparse.ArgumentParser()
parser.add_argument('--config_from_file', default=None, help="configure vae params from file")
parser.add_argument('--epochs', '-e', default = 100, help='Number of training epochs')
# parser.add_argument('--num_chars', '-nc', default=max, help='number of unique SMILES characters')
parser.add_argument('--num_props', default=3, help='number of properties to consider')         
parser.add_argument('--latent_dim', default=100, help='latent dimensions to use')       
parser.add_argument('--batch_size', default=32, help='batch size')         
parser.add_argument('--embedding_dim', default=200, help='embedding dimension')                
parser.add_argument('--n_units', default=96, help='number of hidden rnn units')               
parser.add_argument('--learning_rate', default = 0.002, help='model learning rate')    
parser.add_argument('--kl_rate', default=0.0, help='kl divergence annealing rate')


args = parser.parse_args()

smiles_tokenizer = smiles_data("./data/smiles.txt", 
data_args={'header':None}, tokenizer=tokenizer)

config = Config()
config.update_config(vars(args))

train_dataset = smiles_tokenizer.get_padded_data().batch(config.batch_size)

config.max_seq_len = smiles_tokenizer.max_seq_len
config.num_chars = smiles_tokenizer.num_chars

encoder, decoder, vae = init_vae_models(config.num_chars, config.embedding_dim, config.max_seq_len, config.latent_dim, config.n_units)


losses = []

lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=config.learning_rate,
      decay_steps=config.epochs*len(train_dataset),
      end_learning_rate=0.00001)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
loss_metric = tf.keras.metrics.Mean()
spce_loss = tf.keras.losses.SparseCategoricalCrossentropy()


for epoch in range(config.epochs):

  # iterate over the batches of the dataset.
  for step, batch in enumerate(train_dataset):
    with tf.GradientTape() as tape:
      
      # feed a batch to the VAE model
      reconstructed = vae(batch) # Get a batch of the training examples and feed to the vae model

      loss = spce_loss(batch, reconstructed)  # compute the reconstruction loss between data and reconstruction

      loss += sum(vae.losses)   # add the KL Divergence loss to reconstruction

    grads = tape.gradient(loss, vae.trainable_weights)  # get the gradients with respect to the weights
    optimizer.apply_gradients(zip(grads, vae.trainable_weights)) # Update the weights with gradients

    loss_metric(loss) # compute the mean of losses
    losses.append(loss_metric.result().numpy())
    # # Show outputs at every 50 steps
    if step % 50 == 0:
      print('Epoch: %s step: %s average loss = %s ' % (epoch, step, loss_metric.result().numpy()))


