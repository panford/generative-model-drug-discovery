#@title Utility functions for SMILES property computation
# Define utility functions here
from tensorflow.keras.backend import random_normal
# from rdkit.Chem import Descriptors, QED
import os
import pickle

def config_save(config_path, config_dict):
  # config_dict_path = os.path.join(config_path, 'config.pkl')
  with open(config_path, 'wb') as config_file:
    pickle.dump(config_dict, config_file)

def load_config(config_path):
  # config_dict_path = os.path.join(config_path, 'config.pkl')
  with open(config_path, 'rb') as config_file:
    config_dict = pickle.load(config_file)
  return config_dict



def remove_checkpoints(checkpoint_path):
  for file_name in os.listdir(checkpoint_path):
      # construct full file path
      file = checkpoint_path + file_name
      if os.path.isfile(file):
          print('Deleting file:', file)
          os.remove(file)

def compute_smile_prop(smile):

  """ 
  Compute smiles properties (MolWt, LogP, QED)

  Inputs:
    smile (str, list, tuple) : A sequence or list of sequences of smiles 
                                data whose properties needs to be computed
  Output:
    props (list)  :   Computed properties
  
  """

  def compute_for_one(smi):

    """
    Computes properties for a single smile sequence

    Inputs 
      smi (str) : A sequence of smile characters
    Outputs
      prop (list): Computed properties, "Not exist" if properties cannot be computed
    """

    try:
        mol=Chem.MolFromSmiles(smi) 
        prop = [Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol), QED.qed(mol)]
    except:
        prop = 'Not exist!'
    return prop

      
  if isinstance(smile, (list, tuple)):
    all_list = []
    for s in list(smile):
      all_list.append(compute_for_one(s))
    props = all_list

  elif isinstance(smile, str):
    props = compute_for_one(smile) 
  else:
    print(f"Input must be a string or list, Instead got {type(smile)}")
    
  return props

def canonicalize(smile):
  """Function to canonicalise smiles inputs sequence"""

  return Chem.MolToSmiles(Chem.MolFromSmiles(smile))


def sample_prior(batch_size:int, 
                latent_dim:int):

  """
  Sample prior       :  Sample for random normal distribution
  Inputs:
    batch_size (int) : number of samples to generate
    latent_dim (int) : latent dimension

  Outputs
    samples from normal distribution (size = (batch_size, latent_dim))
  """

  return random_normal((batch_size, latent_dim))