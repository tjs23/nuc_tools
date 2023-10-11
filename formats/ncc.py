import numpy as np
from collections import defaultdict

# #   Globals  # #

#NCC_FORMAT       = '%s %d %d %d %d %s %s %d %d %d %d %s %d %d %d\n'
# See https://github.com/tjs23/nuc_processing/wiki/NCC-data-format

# #   Nuc Formats  # # 

def load_file(file_path, pair_key=True, trans=True, offset=0, dtype=np.uint32, n_max=None, ambig=False):
  """Load chromosome and contact data from NCC format file, as output from NucProcess"""
  
  from core import nuc_io as io
  from core import nuc_util as util
  
  with io.open_file(file_path) as file_obj:
   
    util.info('Reading {}'.format(file_path))
  
    # Observations are treated individually in single-cell Hi-C,
    # i.e. no binning, so num_obs always 1 for each contact
    num_obs = 1
 
    contact_dict = defaultdict(list)
    inactive_min = defaultdict(int)
    inactive_max = defaultdict(int)
    n = 0
    ambig_group = 0
    
    key_counts = {}
    
    for line in file_obj:
      chr_a, f_start_a, f_end_a, start_a, end_a, strand_a, \
      chr_b, f_start_b, f_end_b, start_b, end_b, strand_b, \
      ambig_code, pair_id, swap_pair = line.split()
      
      try:
        ambig_rows, is_active = ambig_code.split('.')
        
        if ambig_rows != '0':
          ambig_group += 1 # Count even if inactive; keep consistent group numbering
      
      except ValueError as err:
        ambig_rows, is_active = '1', '1'
        ambig_group = int(pair_id)
      
      if not ambig and (ambig_rows != '1'):
        continue
      
      if (chr_a != chr_b) and not trans:
        continue

      pos_a = int(f_start_a if strand_a == '+' else f_end_a)
      pos_b = int(f_start_b if strand_b == '+' else f_end_b)
        
      if is_active == '0': # Inactive
        inactive_min[chr_a] = min(pos_a, inactive_min[chr_a] or 1e12)
        inactive_min[chr_b] = min(pos_b, inactive_min[chr_b] or 1e12)
        inactive_max[chr_a] = max(pos_a, inactive_max[chr_a])
        inactive_max[chr_b] = max(pos_b, inactive_max[chr_b])
        continue
        
      if chr_a > chr_b:
        chr_a, chr_b = chr_b, chr_a
        pos_a, pos_b = pos_b, pos_a

      key = (chr_a, chr_b)
      
      if key not in key_counts:
        contact_dict[key] = np.empty((64, 4), dtype)
        key_counts[key] = 0
      
      contact_dict[key][key_counts[key]] = (pos_a, pos_b, num_obs, ambig_group)
      key_counts[key] += 1
      
      if key_counts[key] >= len(contact_dict[key]):
        n_add = min(len(contact_dict[key]), 10000000) # Double initially
        contact_dict[key] = np.concatenate([contact_dict[key], np.empty((n_add, 4), dtype)], axis=0)
            
      n += 1
      
      if n % 100000 == 0:
         util.info(' .. found {:,} contacts'.format(n), line_return =True)
      
  if n_max and (n > n_max):
    util.critical('Too many contacts in ncc file (> %d), this code is meant for single cell data' % n_max)
  else:
    util.info(' .. found {:,} contacts\n'.format(n), line_return =True)
  
  chromo_limits = {}
  contact_dict_out = {}
  keys = sorted(contact_dict)
    
  for key in keys:
    chr_a, chr_b = key

    contact_dict[key] =  contact_dict[key][:key_counts[key]] # Truncate unused allocation
    contacts = contact_dict[key]
      
    seq_pos_a = contacts[:,0]
    seq_pos_b = contacts[:,1]
    
    min_a = min(seq_pos_a)
    max_a = max(seq_pos_a)
    min_b = min(seq_pos_b)
    max_b = max(seq_pos_b)
      
    if chr_a in chromo_limits:
      prev_min, prev_max = chromo_limits[chr_a]
      chromo_limits[chr_a] = [min(prev_min, min_a), max(prev_max, max_a)]
    else:
      chromo_limits[chr_a] = [min_a, max_a]
    
    if chr_b in chromo_limits:
      prev_min, prev_max = chromo_limits[chr_b]
      chromo_limits[chr_b] = [min(prev_min, min_b), max(prev_max, max_b)]
    else:
      chromo_limits[chr_b] = [min_b, max_b]

  if not pair_key:
    pairs = sorted(contact_dict)
    
    for pair in pairs:
      chr_a, chr_b = pair
      if chr_a not in contact_dict:
        contact_dict[chr_a] = {}
        contact_dict[chr_a][chr_b] = contact_dict[pair]
      
      del contact_dict[pair]        
  
  # Find limits for chromos with no active contacts
  for chr_a in inactive_min:
    if chr_a not in chromo_limits:
      chromo_limits[chr_a] = [inactive_min[chr_a], inactive_max[chr_a]]
        
  chromosomes = sorted(chromo_limits)      
        
  return chromosomes, chromo_limits, contact_dict


def getContactMatrix(contact_dict, chrA, chrB, regionA, regionB, binSize=int(1e6)):
  """Gets a full matrix of contacts from sparse contacts""" 
  
  import formats.cyt_ncc

  is_cis = chrA == chrB
    
  startA, endA = regionA
  startB, endB = regionB
  
  s_a = int(startA/binSize)
  s_b = int(startB/binSize)
  e_a = int(endA/binSize)
  e_b = int(endB/binSize)
  
  extentA = endA - startA
  extentB = endB - startB
  
  n = e_a - s_a + 1
  m = e_b - s_b + 1
  
  matrix = np.zeros((n,m), np.int32)
  binSize = np.int32(binSize)
  
  if chrA == chrB:
    chromo_pairs=[(chrA, chrA)]
  else:
    chromo_pairs = [(chrA, chrB), (chrB, chrA)]
    
  for chr1, chr2 in chromo_pairs:
    if chr1 in contact_dict and chr2 in contact_dict[chr1]:
      cData = np.array(contact_dict[chr1][chr2], np.int32)
      if chrA == chr1:
        start1, start2 = np.int32(startA), np.int32(startB)
        transpose = False
        
      else:
        start1, start2 = np.int32(startB), np.int32(startA)
        transpose = True
      
      # Below is additive because both A:B and B:A contacts could be stored
      formats.cyt_ncc.binContacts(cData, matrix, start1, start2,
                  binSize, is_cis, transpose)
  
  return matrix

