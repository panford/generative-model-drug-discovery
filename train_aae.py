import tensorflow.keras as keras
import tensorflow as tf
import argparse
import os
from prepare_data import smiles_data
from tensorflow.keras.preprocessing.text import Tokenizer
from models import init_vae_model, init_aae_model
from config import Config 
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from utils import remove_checkpoints
import pickle
from utils import config_save
from tensorflow.keras.backend import random_normal

tokenizer = Tokenizer(filters=None,
                      lower=False,
                      char_level=True)


parser = argparse.ArgumentParser()
parser.add_argument('--config_from_file', default=None, help="configure vae params from file")
parser.add_argument('--config_file_path', default='./config_dict', help="configure vae params from file")
parser.add_argument('--epochs', '-e', default = 100, help='Number of training epochs')
parser.add_argument('--chkpt_dir', '-cdir', default="./checkpoints", help='directory to save model checkpoints')
parser.add_argument('--num_props', default=3, help='number of properties to consider')         
parser.add_argument('--latent_dim', default=100, help='latent dimensions to use')       
parser.add_argument('--batch_size', default=32, help='batch size')         
parser.add_argument('--embedding_dim', default=200, help='embedding dimension')                
parser.add_argument('--n_units', default=96, help='number of hidden rnn units')               
parser.add_argument('--learning_rate', default = 0.002, help='model learning rate')    
parser.add_argument('--kl_rate', default=0.5, help='kl divergence annealing rate')
parser.add_argument('--model_dir', default='./saved_models', help='directory to save models')
parser.add_argument('--start_from_chkpt','-sfc', default=False, help='begin training from checkpoint')
parser.add_argument('--restart_chkpt','-rc', default=False, help='restart checkpointing')


args = parser.parse_args()

smiles_tokenizer = smiles_data("./data/smiles.txt", 
data_args={'header':None}, tokenizer=tokenizer)

config = Config()
config.update_config(vars(args))

train_dataset = smiles_tokenizer.get_padded_data().batch(config.batch_size)

config.max_seq_len = smiles_tokenizer.max_seq_len
config.num_chars = smiles_tokenizer.num_chars
config.kl_rate = args.kl_rate

adversarial_autoencoder = init_aae_model(config.num_chars, 
                      config.embedding_dim, 
                      config.max_seq_len, 
                      config.latent_dim, 
                      config.n_units,
                      do_variational=True,
                      kl_div_loss_rate=config.kl_rate)


losses = []

lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=config.learning_rate,
      decay_steps=config.epochs*len(train_dataset),
      end_learning_rate=0.00001)

gen_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
gen_enc_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

loss_metric = tf.keras.metrics.Mean()
spce_loss = tf.keras.losses.SparseCategoricalCrossentropy()
bce_loss = tf.keras.losses.BinaryCrossentropy()


model_chkpt = tf.train.Checkpoint(
    model=adversarial_autoencoder, epoch=tf.Variable(0), step=tf.Variable(0), 
    gen_optimizer = gen_optimizer, 
    gen_enc_optimizer = gen_enc_optimizer, 
    disc_optimizer = disc_optimizer, 
    recon_loss= tf.Variable(0.00),
    disc_loss= tf.Variable(0.00),
    gen_loss= tf.Variable(0.00)
    
    )

chkpt_dir = os.path.join(args.chkpt_dir, 'aae_model.ckpt')

config_save(args.config_file_path, vars(config)) #save config

if args.restart_chkpt:
  os.remove(args.chkpt_dir)

chkpt_manager = tf.train.CheckpointManager(
    model_chkpt, directory=chkpt_dir, max_to_keep=1)

last_chkpt = tf.train.latest_checkpoint(chkpt_dir)

if last_chkpt and args.start_from_chkpt:
  model_chkpt.restore(last_chkpt)

  


epochs = 100 # Set the number of training epochs
disc_loss = []
gen_loss = []
avg_recon_loss = []
for epoch in range(epochs):
  # iterate over the batches of the dataset.
  for step, batch in enumerate(train_dataset):

    #  RECONSTRUCTION PHASE
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    with tf.GradientTape() as rtape:
      noise = random_normal((config.batch_size, config.latent_dim))
      adversarial_autoencoder.gen.trainable = True
      # feed a batch to the VAE model
      z, reconstructed = adversarial_autoencoder.gen(batch)                                             # Get a batch of the training examples and feed to the model
    
      recon_loss = spce_loss(batch, reconstructed)                                   # compute the reconstruction loss between data and reconstruction
      # recon_loss += sum(generator.losses)                                          # add the KL Divergence loss to the reconstruction loss for vae
    
    recon_grads = rtape.gradient(recon_loss, adversarial_autoencoder.gen.trainable_weights)          # get the gradients with respect to the weights 
    gen_optimizer.apply_gradients(zip(recon_grads, adversarial_autoencoder.gen.trainable_weights))    # Update the weights with gradients

    loss_metric(recon_loss) # compute the mean of losses


    # REGULARIZATION PHASE
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    # Discriminator Training
    with tf.GradientTape() as dtape:
      noise = random_normal((config.batch_size, config.latent_dim))                                  # Sample from the prior distribution
      # z = encoder(batch)                                                           # Generate the latent distribution with the encoder of the generator
      # concat_data = tf.concat([noise, z], axis =0)                                 # Concatenate both the prior distribution and latent 
      adversarial_autoencoder.disc.trainable = True  
                                                                                     # Set discriminator's weights as trainable 
      z, _ = adversarial_autoencoder.gen(batch)
      batch_pred = adversarial_autoencoder.disc(z)
      batch_labels = tf.zeros_like(batch_pred)

      prior_pred = adversarial_autoencoder.disc(noise)                                              # Predict labels for prior
      prior_labels = tf.ones_like(prior_pred)                                        # Make labels for prior
      batch_loss = bce_loss(batch_labels, batch_pred)                             # compute crossentropy loss for latent
      prior_loss = bce_loss(prior_labels, prior_pred)                                # compute crossentropy loss for prior 

      disc_loss = (batch_loss + prior_loss)/2

    disc_grads = dtape.gradient(disc_loss, adversarial_autoencoder.disc.trainable_weights)          # Compute gradients wrt discriminator weights
    disc_optimizer.apply_gradients(zip(disc_grads, adversarial_autoencoder.disc.trainable_weights)) # Update weights

    
    # Generator training
    with tf.GradientTape() as gtape:

      adversarial_autoencoder.disc.trainable = False                                                # Set discriminator weights to not trainable  
      z,_ = adversarial_autoencoder.gen(batch) 
      batch_pred = adversarial_autoencoder.disc(z)                                            # Predict labels for noise
      labels = tf.ones_like(batch_pred)                                              # Make labels to fool discriminator
      gen_loss = bce_loss(labels, batch_pred)                                        # Get loss from discriminator

    gen_grads = gtape.gradient(gen_loss, adversarial_autoencoder.gen.encoder.trainable_weights)
    gen_enc_optimizer.apply_gradients(zip(gen_grads, adversarial_autoencoder.gen.encoder.trainable_weights))


    # Show outputs at every 50 steps
    if step % 50 == 0:
      print('Epoch: %s\t step: %s \n average recon loss: %s \t disc loss: %s \t gen loss: %s' % (epoch, step, loss_metric.result().numpy(), disc_loss.numpy(), gen_loss.numpy()))
      model_chkpt.step.assign(step)
      model_chkpt.recon_loss.assign(recon_loss)
      model_chkpt.disc_loss.assign(disc_loss)
      model_chkpt.gen_loss.assign(gen_loss)
      model_chkpt.epoch.assign(epoch)
      # save_path = chkpt_manager.save()

  
aae_model_dir = os.path.join(args.model_dir, 'aae_model.h5')
adversarial_autoencoder.save(aae_model_dir)
