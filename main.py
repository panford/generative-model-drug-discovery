import tensorflow.keras as keras
import tensorflow as tf
import argparse
import os
from prepare_data import smiles_data
from tensorflow.keras.preprocessing.text import Tokenizer
from models import init_vae_model
from config import Config 
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from utils import remove_checkpoints
import pickle
from utils import config_save

tokenizer = Tokenizer(filters=None,
                      lower=False,
                      char_level=True)

def train_vae(config, train_dataset, vae, loss_fn, optimizer, model_chkpt, chkpt_manager, loss_metric):
  for epoch in range(config.epochs):

  # iterate over the batches of the dataset.
    for step, batch in enumerate(train_dataset):
      with tf.GradientTape() as tape:


        _, reconstructed = vae(batch) 

        loss = loss_fn(batch, reconstructed) 
        
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


def train_aae():

  pass
    


  
vae_model_dir = os.path.join(args.model_dir, 'vae_model.h5')
vae.save(vae_model_dir)
