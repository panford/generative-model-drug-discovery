import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences


class smiles_data():
  def __init__(self, data_path, data_args, tokenizer):
    self.data_path = data_path
    self.tokenizer = tokenizer
    self.data = pd.read_csv(data_path, **data_args)[0]
    self.fitted_on_data = False
    self.max_seq_len = None
    self.num_chars = None 

  def fit_tokenizer(self):
    self.tokenizer.fit_on_texts(self.data)
    self.num_chars = len(self.tokenizer.word_index)+1
    self.fitted_on_data = True

  def as_tensor_dataset_(self, data):
      return Dataset.from_tensor_slices(data)

  def get_padded_data(self, as_tensor_dataset=True):
    """
    Creates numerical dataset from character sequences and pads up to the max sequence length
    Inputs
      smiles_(list):  lists of sequences from data
    Outputs:
      (nd.array):     An array of padded sequences up to max sequence length
      max_seq_len (int): length of longest sequence in the data

    """
    if self.fitted_on_data is not True:
      self.fit_tokenizer()

    x_sequences = [] 
    
    # Loop through each row
    for line in self.data:
      token_list = self.tokenizer.texts_to_sequences([line])[0] #Tokenize each row
      x_sequences.append(token_list) # append to x_sequences

    # pad sequences 
    self.max_seq_len = max([len(x) for x in x_sequences]) # Compute max sequence length

    padded_data = np.array(pad_sequences(x_sequences, maxlen=self.max_seq_len, padding='pre'))

    if as_tensor_dataset is True:
      padded_data = self.as_tensor_dataset_(padded_data)

    return padded_data

    

 




