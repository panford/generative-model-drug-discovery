import tensorflow.keras as keras
import tensorflow as tf
import argparse
import os
import tensorboard
from prepare_data import SMILES
from tensorflow.keras.preprocessing.text import Tokenizer
from models import init_vae_model
from config import VaeConfig, TrainingConfig, PathsConfig
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from utils import remove_checkpoints #, config_save
import pickle

tokenizer = Tokenizer(filters=None,
                      lower=False,
                      char_level=True)


parser = argparse.ArgumentParser()
parser.add_argument('--config_from_file', default=None, help="configure vae params from file")
parser.add_argument('--config_file_path', default='./config_dict', help="configure vae params from file")
parser.add_argument('--epochs', '-e', default = 50, help='Number of training epochs')
parser.add_argument('--checkpoint_dir', '-cdir', default="./vae_checkpoints", help='directory to save model checkpoints')
parser.add_argument('--num_props', default=3, help='number of properties to consider')         
parser.add_argument('--latent_dim', default=100, help='latent dimensions to use')       
parser.add_argument('--batch_size', default=32, help='batch size')         
parser.add_argument('--embedding_dim','-edim', default=200, help='embedding dimension')                
parser.add_argument('--n_units', default=96, help='number of hidden rnn units')               
parser.add_argument('--learning_rate', '-lr', default = 0.002, help='model learning rate')    
parser.add_argument('--kl_div_loss_rate', default=0.5, help='kl divergence annealing rate')
parser.add_argument('--save_model_dir', '-mdir', default='./saved_models', help='directory to save models')
parser.add_argument('--start_from_chkpt','-sfc', default=False, help='begin training from checkpoint')
parser.add_argument('--restart_chkpt','-rc', default=True, help='restart checkpointing')
parser.add_argument('--data_path', default="./data/smiles.txt", help="data directory")


args = parser.parse_args()

vae_config = VaeConfig()
paths_config = PathsConfig()
train_config = TrainingConfig()

vae_config.update(args)
train_config.update(args)
paths_config.update(args)


smiles_tokenizer = SMILES(paths_config.data_path, data_args={'header':None}, tokenizer=tokenizer)
train_dataset = smiles_tokenizer.get_padded_data().batch(train_config.batch_size)

vae_config.max_seq_len = smiles_tokenizer.max_seq_len
vae_config.num_chars = smiles_tokenizer.num_chars
# config.kl_rate = args.kl_rate

vae = init_vae_model(**vars(vae_config))

# print("model configs: ", vae_config)
losses = []

lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=train_config.learning_rate,
      decay_steps=train_config.epochs*len(train_dataset),
      end_learning_rate=0.00001)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
loss_metric = tf.keras.metrics.Mean()
spce_loss = tf.keras.losses.SparseCategoricalCrossentropy()


model_chkpt = tf.train.Checkpoint(
    model=vae, epoch=tf.Variable(0), step=tf.Variable(0), optimizer = optimizer, loss= tf.Variable(0.00))

paths_config.checkpoint_file = os.path.join(paths_config.checkpoint_dir, 'vae_model.ckpt')

if args.restart_chkpt:
  remove_checkpoints(paths_config.checkpoint_file)

chkpt_manager = tf.train.CheckpointManager(
    model_chkpt, directory=paths_config.checkpoint_file, max_to_keep=1)

last_chkpt = tf.train.latest_checkpoint(paths_config.checkpoint_file)

if last_chkpt and args.start_from_chkpt:
  model_chkpt.restore(last_chkpt)

print("*"*15)
print("Training Begun")
print("*"*15)


# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)


for epoch in range(train_config.epochs):

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

  # log_results(loss, epoch)

  if epoch %50 == 0:
    model_chkpt.step.assign(step)
    model_chkpt.loss.assign(loss)
    model_chkpt.epoch.assign(epoch)
    save_path = chkpt_manager.save()

  spce_loss.reset_states()
  loss_metric.reset_states()


vae_config.save(args)
train_config.save(args)
paths_config.save(args)
vae_model_dir = os.path.join(paths_config.save_model_dir, 'vae_model.h5')
vae.save(vae_model_dir)
