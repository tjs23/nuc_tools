
import numpy as np

def load_npz_contacts(file_path):
  
  file_dict = np.load(file_path)
  
  chromo_limits = {}
  contacts = {}
  bin_size, min_bins = file_dict['params']
  bin_size = int(bin_size*1e3)
  
  for key in file_dict:
    if key != 'params':
      if ' ' in key:
        chr_a, chr_b = key.split()
        contacts[(chr_a, chr_b)] = file_dict[key][()].toarray()
      else:
        offset, count = file_dict[key]
        chromo_limits[key] = offset * bin_size, (offset + count) * bin_size
  
  return bin_size, chromo_limits, contacts

