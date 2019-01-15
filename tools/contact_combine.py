import os, sys, math
import numpy as np
from collections import defaultdict 
from scipy import sparse

PROG_NAME = 'contact_combine'
VERSION = '1.0.0'
DESCRIPTION = 'Combine two Hi-C contact maps (NPZ format)'

def contact_combine(in_path_1, in_path_2, out_path=None, store_sparse=True): 
    
  from nuc_tools import util, io
  from formats import npz
  
  if not out_path:
    out_path = io.merge_file_names(in_path_a, in_path_b, prefix='comb_')
    
  if not out_path.endswith('.npz'):
    out_path = out_path + '.npz'
  
  file_bin_size_1, chromo_limits_1, contacts_1 = npz.load_npz_contacts(in_path_1)
  file_bin_size_2, chromo_limits_2, contacts_2 = npz.load_npz_contacts(in_path_2)

  if file_bin_size_1 != file_bin_size_1:
    util.critical('Chromatin contact matrices to be compared must be binned at the same resolution')
    # Above could be relaxed as long as one is a multiple of the other, and lowest resolution is used
  
  util.info('Combining {} with {}'.format(in_path_1, in_path_2))  
  
  bin_size = file_bin_size_1
  chromos = list(set(chromo_limits_1.keys()) & set(chromo_limits_2.keys()))
  chromos = util.sort_chromosomes(chromos)

  comb_limits = {}
  
  for chromo in chromos:
    s1, e1 = chromo_limits_1[chromo]
    s2, e2 = chromo_limits_2[chromo]
    comb_limits[chromo] = (0, max(e1, e2))
  
  for k, chr_a in enumerate(chromos):
    n_a = int(math.ceil(comb_limits[chr_a][1]/bin_size))
    p_a = int(s1/bin_size)
    
    for chr_b in chromos[k:]:
      pair = (chr_a, chr_b)
      
      if (pair not in contacts_1) and (pair not in contacts_2):
        continue
        
      util.info(' .. working on {}:{}'.format(chr_a, chr_b), line_return=True)  
      
      is_cis = chr_a == chr_b
      n_b = int(math.ceil(comb_limits[chr_b][1]/bin_size)) 
      mat = np.zeros((n_a, n_b), 'uint32')
              
      if pair in contacts_1:
        n, m = contacts_1[pair].shape
        pa = int(chromo_limits_1[chr_a][0]/bin_size)
        pb = int(chromo_limits_1[chr_b][0]/bin_size)
        mat[pa:pa+n,pb:pb+m] += contacts_1[pair]
        
      if pair in contacts_2:
        n, m = contacts_2[pair].shape
        pa = int(chromo_limits_2[chr_a][0]/bin_size)
        pb = int(chromo_limits_2[chr_b][0]/bin_size)
        mat[pa:pa+n,pb:pb+m] += contacts_2[pair]
        
      if store_sparse:
        if is_cis:
          mat = sparse.csr_matrix(mat)
        else:
          mat = sparse.coo_matrix(mat)
      
      contacts_1[pair] = mat # Overwite to save memory

  util.info('Saving data')
    
  npz.save_contacts(out_path, contacts_1, comb_limits, bin_size, min_bins=0)  
  
  util.info('Written NPZ file {}'.format(out_path))
  
  
def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)

  arg_parse.add_argument(metavar='CONTACT_FILES', nargs=2, dest='i',
                         help='Two input NPZ format (binned, bulk Hi-C data) chromatin contact files to be combined.')

  arg_parse.add_argument('-o', metavar='OUT_FILE', default=None,
                         help='Optional output NPZ format file name. If not specified, a default based on the input file names will be used.')
 
  args = vars(arg_parse.parse_args(argv))

  in_path_a, in_path_b = args['i']
  out_path = args['o']

  invalid_msg = io.check_invalid_file(in_path_a)
  if invalid_msg:
    util.critical(invalid_msg)

  invalid_msg = io.check_invalid_file(in_path_b)
  if invalid_msg:
    util.critical(invalid_msg)
  
  if io.is_same_file(in_path_a, in_path_b):
    util.warn('Inputs being compared are the same file')  
    
  contact_combine(in_path_a, in_path_b, out_path)
  

if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()
