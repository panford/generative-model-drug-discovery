## Generative Models for Drug Discovery
Kobby Panford-Quainoo

1.This is my implementation of [Variational Autoencoders](https://arxiv.org/abs/1312.6114) and [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644) for generating drug sequences (SMILES)

2. For questions and issues, please email panfordkobby@gmail.com

## Using this repo
1. Clone the repository
```sh
 $ git clone https://github.com/panford/generative-model-drug-discovery.git
```
2. Install dependencies from the requirement file:
 ```sh
   $ pip install -r requirements.txt
   ```
 3. Train model: ```train.py``` can be run now and arguments may be passed in the terminal or passed as a text file
 ```sh
   $ python train.py -e 50
   ```
 5. Generate some smiles :), example (using temperature sampling with temperature value of 0.5 : 
  ```sh
  $ python generate.py --decoding_strategy "temperature" --decoder_params 0.5
  ```
  




