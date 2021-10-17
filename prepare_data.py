import tensorflow as tf

smiles_data = pd.read_csv(path_to_data, header = None)
smiles_data = smiles_data[0][:2000]