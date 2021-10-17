#@title Utility functions for decoding sequences
def greedy_search():
  """
  A decoder algorithm to retrieve sequences from decoder output

  Inputs
    inputs (tf-tensor): Output from decoder, size (batch, max_seq_len, num_chars)
  
  Outputs
    seq of greedily decoded sequences
  """
  def decode(preds):
    return np.argmax(preds).tolist()

  return decode



def temperature_sampling(temperature):
  """
  Temperature sampling wrapper function

  This wrapper function will allow us use the temperature sampling strategy to decode our predicted sequences
  """
  def softmax(z):
    """Softmax function """
    return np.exp(z)/sum(np.exp(z))

  def decode(preds):

    """ 
    Decoder using temperature 
    """

    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    reweighted_preds = softmax(preds)
    probs = np.random.multinomial(1, reweighted_preds, 1)

    return np.argmax(probs)

  return decode



def decode_preds(preds, d_strategy="temp", **decoder_params):

  """Decoding function: call this with decoder prediction
  Inputs
    preds (batch_size, max_seq_len, num_chars): softmax prediction from decoder model
    d_strategy (str):      Strategy to use for prediction
  """

  
  if d_strategy == "temp":
    temperature = decoder_params.get('temperature') or 0.5
    strategy = temperature_sampling(temperature)    
    print('Decoding strategy: ', "Temperature Sampling (%s) "%temperature)

  elif d_strategy == 'greedy':
    strategy = greedy_search()
    print('Decoding strategy: ', "Greedy Search")
  print("*"*8)

  seqs = []
  for n in range(preds.shape[0]):
    batch = []
    for l in range(preds.shape[1]):
      batch.append(strategy(preds[n,l,:]))
    seqs.append(batch)
  decoded_seq = tokenizer.sequences_to_texts(unpad(list(seqs)))
  seq = [s.replace(" ","") for s in decoded_seq]

  return seq


def generate_smiles_from_prior(model, prior, decoding_strategy = "temp", **decoder_params):
  
  """
  Generates smiles samples from prior

  Inputs
    batch_size: Batch size of samples to generate
    latent_dim: Latent dimension 

  outputs
    decoded_seq: Decoded sequence

  """

  predicted_seq = model.predict(prior)

  return decode_preds(predicted_seq, decoding_strategy, **decoder_params)


def unpad(input_tokens):
  """Function for unpadding tokens
  Inputs
    input_tokens  : list of input tokens
  Outputs
    unpadded tokens (list)
  """

  unpadded = []
  for i in range(len(input_tokens)):
    unpadded_list = [token for token in input_tokens[i] if token !=0]
    unpadded.append(unpadded_list)

  return unpadded


def generate_random_smiles(model, batch_size, latent_dim, decoding_strategy, **decoder_params):
  
  """
  Generates smiles samples from random normal samples

  Inputs
    batch_size: Batch size of samples to generate
    latent_dim: Latent dimension 
    model     : model to predict from prior

  outputs
    decoded_seq: Decoded sequence

  """

  prior = tf.random.normal(shape=[batch_size, latent_dim],)

  return generate_smiles_from_prior(model, prior, decoding_strategy, **decoder_params)