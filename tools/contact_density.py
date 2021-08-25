import sys,  os, math
import numpy as np


PROG_NAME = 'contact_density'
VERSION = '1.0.0'
DESCRIPTION = 'Creates linear contact density data tracks, in BED format, from 2D Hi-C data for different sequence separation ranges and in trans'
SEQ_SEP_THRESHOLDS = [10,100,1000]
DEFAULT_BIN_SIZE = 25.0


# Ad an all cis counts track cis

def contact_density(contact_path, out_path_root=None, bin_size=DEFAULT_BIN_SIZE, seq_sep_thresholds=SEQ_SEP_THRESHOLDS, use_trans=False):

  from nuc_tools import util, io
  from formats import ncc, npz, bed
  
  seq_sep_thresholds = sorted([int(1e3 * x) for x in seq_sep_thresholds])
    
  if  out_path_root:
    out_path_root = os.path.splitext(out_path_root)[0]
  else:
    out_path_root = os.path.splitext(contact_path)[0] + '_contact_density'
  
  
  if io.is_ncc(contact_path):
    file_bin_size = None
    bin_size = int(bin_size * 1e3)
    util.info('  .. loading NCC contacts {}'.format(contact_path))
    chromosomes, chromo_limits, contacts = ncc.load_file(contact_path, trans=use_trans, dtype=np.int32)
    
  else:
    util.info('  .. loading NPZ contacts {}'.format(contact_path))
    file_bin_size, chromo_limits, contacts = npz.load_npz_contacts(contact_path, trans=use_trans)
    bin_size = file_bin_size
    chromosomes = chromo_limits.keys()
  
  chromosomes = util.sort_chromosomes(chromosomes)
  
  n_tracks = len(seq_sep_thresholds) + 2 # Add very big, plus all ciss
  
  if use_trans:
    n_tracks += 1
  
  data_tracks = []
  
  for i in range(n_tracks):
    data_dict = {}
    
    for chromo in chromosomes:
      start, end = chromo_limits[chromo] # starts might not be on-bin
      n_bins = int(math.ceil(end/float(bin_size)))
      data_dict[chromo] = np.zeros(n_bins)
  
    data_tracks.append(data_dict)
  
  thr_max_min = [(seq_sep_thresholds[0], None)]
  
  for i, th in enumerate(seq_sep_thresholds[1:], 1):
    thr_max_min.append((th, seq_sep_thresholds[i-1]))
  
  thr_max_min.append((None, seq_sep_thresholds[-1]))
  thr_max_min.append((None, None))
                   
  track_tags = []
  for mx, mn in thr_max_min:
    if mx and mn:
      tag = '%d-%dk' % (mn/1e3, mx/1e3)
    elif mx:
      tag = '0-%dk' % (mx/1e3,)
    elif mn:
      tag = '%dk+' % (mn/1e3)
    else:
      tag = 'all'
      
    track_tags.append(tag)
  
  if use_trans:  
    track_tags.append('trans')  
  
  for chr_a, chr_b in contacts:
    chromo_pair= (chr_a, chr_b)
    n_a = len(data_tracks[0][chr_a])
    n_b = len(data_tracks[0][chr_b])
    
    if file_bin_size:
      matrix = contacts[chromo_pair]
      
      if chr_a == chr_b: # cis
        n = len(matrix)
        data_idx = 0
 
        for d in range(0, n): # diagonal line
          m = n-d
          rows = np.array(range(m))
          cols = rows + d
          idx = (rows, cols)
          seq_sep = d * file_bin_size
       
          if d and (seq_sep >= seq_sep_thresholds[data_idx]) and ((d-1) * file_bin_size < seq_sep_thresholds[data_idx]):
            data_idx += 1
          
          data_dict = data_tracks[data_idx]
          data_dict[chr_a][:m] += matrix[idx] # rows
          data_dict[chr_a][d:] += matrix[idx] # cols
        
        # all cis
        data_dict = data_tracks[data_idx+1]
        data_dict[chr_a] += matrix.sum(axis=1)
        data_dict[chr_b] += matrix.sum(axis=0)        
        
      elif use_trans:
        data_dict_all   = data_tracks[-2]
        data_dict_trans = data_tracks[-1]
        msum = matrix.sum(axis=1)
        
        data_dict_all[chr_a] += msum
        data_dict_trans[chr_a] += msum

        msum = matrix.sum(axis=0)
        data_dict_all[chr_b] += msum
        data_dict_trans[chr_b] += msum
     
    else:
      contact_array = contacts[chromo_pair]
      
      seq_pos_a = contact_array[:,0]
      seq_pos_b = contact_array[:,1]
      chromo_counts = contact_array[:,2]
      seps = abs(seq_pos_a-seq_pos_b)
      
      bins_a = (seq_pos_a/bin_size).astype(int)
      bins_b = (seq_pos_b/bin_size).astype(int)
      
      if chr_a == chr_b:
        for i, (th_max, th_min) in enumerate(thr_max_min): # Short/long range seq separation limits
          if th_max and th_min:
            idx = (seps < th_max) & (seps >= th_min)
          elif th_max:
            idx = seps < th_max
          elif th_min:
            idx = seps >= th_min
          else:
            idx = seps > 0
          
          bin_counts_a, null = np.histogram(bins_a[idx], bins=n_a, range=(0,n_a))
          bin_counts_b, null = np.histogram(bins_b[idx], bins=n_b, range=(0,n_b))
          data_dict = data_tracks[i]
          data_dict[chr_a] += bin_counts_a
          data_dict[chr_a] += bin_counts_b
        
      elif use_trans:
        bin_counts_a, null = np.histogram(bins_a, bins=n_a, range=(0,n_a))
        bin_counts_b, null = np.histogram(bins_b, bins=n_b, range=(0,n_b))
        
        data_dict = data_tracks[-1] # Trans only
        data_dict[chr_a] += bin_counts_a
        data_dict[chr_b] += bin_counts_b

        data_dict = data_tracks[-2] # All
        data_dict[chr_a] += bin_counts_a
        data_dict[chr_b] += bin_counts_b
  
  for i, data_dict in enumerate(data_tracks):
    out_file_path = '{}_{}.bed'.format(out_path_root, track_tags[i])
    total = 0
    
    for chromo in data_dict:
      data_dict[chromo] = util.hist_to_data_track(data_dict[chromo], bin_size)
      total += len(data_dict[chromo])

    if total:
      bed.save_data_track(out_file_path, data_dict, scale=1.0, as_float=False)
      util.info('Written {} regions to {}'.format(total, out_file_path))   
    
    
def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)

  arg_parse.add_argument(metavar='CONTACT_FILE', dest='i',
                         help='Input NPZ or NCC format chromatin contact file.')

  arg_parse.add_argument('-s', '--bin-size', default=DEFAULT_BIN_SIZE, metavar='BIN_SIZE', type=float, dest="s",
                         help='For NCC format input, the binned region size for output data track, in kilobases. ' \
                              'Defaults to %.1f kb but totally ignored if NPZ data is input: here the native bin size of the contact map is used.' % DEFAULT_BIN_SIZE)

  arg_parse.add_argument('-o', '--out-file-root',metavar='OUT_FILE_ROOT', dest='o',
                         help='Optional file path to specify where output data track files will be saved.' \
                              'All output file names will be appended with the different sequence separation thresholds.')
  
  thresh_text = '{} {} {}'.format(*SEQ_SEP_THRESHOLDS)
  
  arg_parse.add_argument('-k', '--kb-thresh', metavar='KB_THRESHOLD', nargs='*', dest="k", type=float, 
                         help='Sequence separation thresholds (in kb) for stratifying contact density. Default: %s' % thresh_text)

  arg_parse.add_argument('-t', '--trans', default=False, action='store_true', dest="t",
                         help='Also, consider inter-chromosomal (trans) contacts as a separate category')
 
  args = vars(arg_parse.parse_args(argv))

  contact_path = args['i']
  out_path_root = args['o']
  bin_size = args['s']
  thresholds = args['k']
  use_trans = args['t']
  
  io.check_invalid_file(contact_path)  
  
  contact_density(contact_path, out_path_root, bin_size, thresholds, use_trans)
  

if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()

