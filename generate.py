import tensorflow as tf
import numpy as np
import argparse
from decoders import *
import pickle
from utils import load_config
import os



parser = argparse.ArgumentParser()
parser.add_argument('--decoding_strategy','-sm', default='temperature', help='decoding strategy to use')
parser.add_argument('--decoder_params', '-bm', default=5, help='parameters for the chosen decoder')
parser.add_argument('--model_dir', default='./saved_models', help='directory to save models')
parser.add_argument('--config_file_path', default='./config_dict', help="configure vae params from file")

args = parser.parse_args()

config = load_config(args.config_file_path)
vae = tf.keras.models.load_model(os.path.join(args.model_dir, 'vae_model.h5'))
generated_smiles = generate_random_smiles(vae, batch_size, config.latent_dim, args.decoding_strategy, args.decoder_params)

print(generated_smiles)