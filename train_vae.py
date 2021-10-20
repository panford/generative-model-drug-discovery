import tensorflow.keras as keras
import tensorflow as tf
import argparse
import os
from prepare_data import smiles_data
from tensorflow.keras.preprocessing.text import Tokenizer
from models import init_vae_models
from config import Config 
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from utils import remove_checkpoints
import pickle
from utils import config_save

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

vae = init_vae_models(config.num_chars, 
                      config.embedding_dim, 
                      config.max_seq_len, 
                      config.latent_dim, 
                      config.n_units,
                      config.kl_rate)


losses = []

lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=config.learning_rate,
      decay_steps=config.epochs*len(train_dataset),
      end_learning_rate=0.00001)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
loss_metric = tf.keras.metrics.Mean()
spce_loss = tf.keras.losses.SparseCategoricalCrossentropy()


model_chkpt = tf.train.Checkpoint(
    model=vae, epoch=tf.Variable(0), step=tf.Variable(0), optimizer = optimizer, loss= tf.Variable(0.00))

chkpt_dir = os.path.join(args.chkpt_dir, 'vae_model.ckpt')

config_save(args.config_file_path, vars(config)) #save config

if args.restart_chkpt:
  os.remove(args.chkpt_dir)

chkpt_manager = tf.train.CheckpointManager(
    model_chkpt, directory=chkpt_dir, max_to_keep=1)

last_chkpt = tf.train.latest_checkpoint(chkpt_dir)

if last_chkpt and args.start_from_chkpt:
  model_chkpt.restore(last_chkpt)

  
for epoch in range(config.epochs):

  # iterate over the batches of the dataset.
  for step, batch in enumerate(train_dataset):
    with tf.GradientTape() as tape:


      _, reconstructed = vae(batch) 

      loss = spce_loss(batch, reconstructed) 
      
      loss += vae.kl_loss 

    grads = tape.gradient(loss, vae.trainable_weights)  
    optimizer.apply_gradients(zip(grads, vae.trainable_weights)) 

    loss_metric(loss) 
    losses.append(loss_metric.result().numpy())

    if step % 50 == 0:
      print('Epoch: %s step: %s average loss = %s ' % (epoch, step, loss_metric.result().numpy()))

      model_chkpt.step.assign(step)
      model_chkpt.loss.assign(loss)
      model_chkpt.epoch.assign(epoch)
      save_path = chkpt_manager.save()

  
vae_model_dir = os.path.join(args.model_dir, 'vae_model.h5')
vae.save(vae_model_dir)
