"""
Feature calculations.
"""
import logging
import types
import numpy as np
import multiprocessing
###################################################
import logging
logging.basicConfig(level=logging.DEBUG)
###################################################

def _featurize_complex(featurizer, mol_pdb_file, protein_pdb_file, log_message):
  logging.info(log_message)
  return featurizer._featurize_complex(mol_pdb_file, protein_pdb_file)


class ComplexFeaturizer(object):
  """"
  Abstract class for calculating features for mol/protein complexes.
  """

  def featurize_complexes(self, mol_files, protein_pdbs, parallelize=True):
    """
    Calculate features for mol/protein complexes.

    Parameters
    ----------
    mols: list
      List of PDB filenames for molecules.
    protein_pdbs: list
      List of PDB filenames for proteins.
    parallelize: bool
      Use multiprocessing to parallelize

    Returns
    -------
    features: np.array
      Array of features
    failures: list
      Indices of complexes that failed to featurize.
    """
    if parallelize:
      pool = multiprocessing.Pool()
      results = []
      for i, (mol_file, protein_pdb) in enumerate(zip(mol_files, protein_pdbs)):
        log_message = "Featurizing %d / %d" % (i, len(mol_files))
        results.append(
            pool.apply_async(_featurize_complex,
                             (self, mol_file, protein_pdb, log_message)))
      pool.close()
    else:
      results = []
      for i, (mol_file, protein_pdb) in enumerate(zip(mol_files, protein_pdbs)):
        log_message = "Featurizing %d / %d" % (i, len(mol_files))
        results.append(
            _featurize_complex(self, mol_file, protein_pdb, log_message))
    features = []
    failures = []
    for ind, result in enumerate(results):
      if parallelize:
        new_features = result.get()
      else:
        new_features = result
      # Handle loading failures which return None
      if new_features is not None:
        features.append(new_features)
      else:
        failures.append(ind)
    features = np.asarray(features)
    return features, failures

  def _featurize_complex(self, mol_pdb, complex_pdb):
    """
    Calculate features for single mol/protein complex.

    Parameters
    ----------
    mol_pdb: list
      Should be a list of lines of the PDB file.
    complex_pdb: list
      Should be a list of lines of the PDB file.
    """
    raise NotImplementedError('Featurizer is not defined.')


class Featurizer(object):
  """
  Abstract class for calculating a set of features for a molecule.

  Child classes implement the _featurize method for calculating features
  for a single molecule.
  """

  def featurize(self, mols, verbose=True, log_every_n=1000):
    """
    Calculate features for molecules.

    Parameters
    ----------
    mols : iterable
        RDKit Mol objects.
    """
    mols = list(mols)
    features = []
    for i, mol in enumerate(mols):
      if mol is not None:
        features.append(self._featurize(mol))
      else:
        features.append(np.array([]))

    features = np.asarray(features)
    return features

  def _featurize(self, mol):
    """
    Calculate features for a single molecule.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    raise NotImplementedError('Featurizer is not defined.')

  def __call__(self, mols):
    """
    Calculate features for molecules.

    Parameters
    ----------
    mols : iterable
        RDKit Mol objects.
    """
    return self.featurize(mols)


class UserDefinedFeaturizer(Featurizer):
  """Directs usage of user-computed featurizations."""

  def __init__(self, feature_fields):
    """Creates user-defined-featurizer."""
    self.feature_fields = feature_fields
