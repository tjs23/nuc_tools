import math
import numpy as np
from scipy import sparse

CHR_KEY_SEP = ' '

def get_chromosomes(file_path):
  
  chromos = set()
  
  file_dict = np.load(file_path, allow_pickle=True)
 
  for key in sorted(file_dict):
    if key != 'params':
      if CHR_KEY_SEP in key:
        chr_a, chr_b = key.split(CHR_KEY_SEP)
        chromos.add(chr_a)
        chromos.add(chr_b)
  
  return chromos
  
  
def load_npz_contacts(file_path, trans=True, store_sparse=False, display_counts=False):
  
  file_dict = np.load(file_path, allow_pickle=True, encoding='latin1')
  
  chromo_limits = {}
  contacts = {}
  bin_size, min_bins = file_dict['params']
  bin_size = int(bin_size*1e3)
  
  chromo_hists = {}
  cis_chromo_hists = {}
  
  for key in sorted(file_dict):
    if key != 'params':
      if CHR_KEY_SEP in key:
        chr_a, chr_b = key.split(CHR_KEY_SEP)
        
        if (chr_a == chr_b) or trans:
          
          try:
            mat = file_dict[key][()]
          except UnicodeError as err:
            print(err)
            print('*'*25)
            print(key, chr_a, chr_b)
            continue
            
          if not store_sparse:
             mat = mat.toarray()
 
          if chr_a == chr_b:
            a, b = mat.shape
 
            if a != b:
              a = min(a,b)
              mat = mat[:a,:a]
 
            cols = np.arange(a-1)
            rows = cols-1

            if not np.all(mat[rows, cols] == mat[cols, rows]): # Not symmetric
              mat += mat.T
 
          contacts[(chr_a, chr_b)] = mat                                                
          
   
      else:
        offset, count = file_dict[key]
        chromo_limits[key] = offset * bin_size, (offset + count) * bin_size
        chromo_hists[key] = np.zeros(count)
        cis_chromo_hists[key] = np.zeros(count)
  
  if display_counts:
    # A simple 1D overview of count densities
 
    from matplotlib import pyplot as plt

    for chr_a, chr_b in contacts:
      mat = contacts[(chr_a, chr_b)]
      chromo_hists[chr_a] += mat.sum(axis=1)
      chromo_hists[chr_b] += mat.sum(axis=0)
 
      if chr_a == chr_b:
        cis_chromo_hists[chr_a] += mat.sum(axis=1)
        cis_chromo_hists[chr_b] += mat.sum(axis=0)
    
    all_sums = np.concatenate([chromo_hists[ch] for ch in chromo_hists])
    cis_sums = np.concatenate([cis_chromo_hists[ch] for ch in chromo_hists])
 
    fig, ax = plt.subplots()
 
    hist, edges = np.histogram(all_sums, bins=25, normed=False, range=(0, 500))
    ax.plot(edges[1:], hist, color='#0080FF', alpha=0.5, label='Whole genome (median=%d)' % np.median(all_sums))

    hist, edges = np.histogram(cis_sums, bins=25, normed=False, range=(0, 500))
    ax.plot(edges[1:], hist, color='#FF4000', alpha=0.5, label='Intra-chromo/contig (median=%d)' % np.median(cis_sums))
 
    ax.set_xlabel('Number of Hi-C RE fragment ends (%d kb region)' % (bin_size/1e3))
    ax.set_ylabel('Count')
 
    ax.legend()
 
    plt.show()

  
  return bin_size, chromo_limits, contacts


def save_contacts(out_file_path, matrix_dict, chromo_limits, bin_size, min_bins=0):
  
  contacts = {}
  kb_bin_size = int(bin_size/1e3)
  
  for chr_a, chr_b in matrix_dict:
    pair = chr_a, chr_b
    key = CHR_KEY_SEP.join(pair)
  
    if chr_a == chr_b:
      contacts[key] = sparse.csr_matrix(matrix_dict[pair])
    else:
      contacts[key] = sparse.coo_matrix(matrix_dict[pair])
    
    start_a, end_a = chromo_limits[chr_a]
    start_b, end_b = chromo_limits[chr_b]
    
    min_a = int(start_a/bin_size)
    num_a = int(math.ceil(end_a/bin_size)) - min_a
    min_b = int(start_b/bin_size)
    num_b = int(math.ceil(end_b/bin_size)) - min_b
    
    # Store bin offsets and spans
    contacts[chr_a] = np.array([min_a, num_a])
    contacts[chr_b] = np.array([min_b, num_b])
    
    contacts['params'] = np.array([kb_bin_size, min_bins])    
  
  np.savez_compressed(out_file_path, **contacts)    
  
