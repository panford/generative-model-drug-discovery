import tensorflow.keras as keras
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-e', default = 100, help='Number of training epochs')
parser.add_argument('--num_chars', '-nc', default=max, help='number of unique SMILES characters')

args = parser.parse_args()

encoder = encoder_model(args.num_chars, args.embedding_dim, max_seq_len, latent_dim, n_units)
decoder = decoder_model(num_chars, max_seq_len, latent_dim, n_units)
vae = vae_model(encoder, decoder, max_seq_len)
# prior = sample_prior(1, latent_dim)
losses = []
for epoch in range(epochs):

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