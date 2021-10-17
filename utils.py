#@title Utility functions for SMILES property computation
# Define utility functions here
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