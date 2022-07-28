import sys, os, math, time
import numpy as np

from collections import defaultdict
from scipy import sparse
from sys import stdout

PROG_NAME = 'ncc_bin'
VERSION = '1.0.0'
DESCRIPTION = 'Perform uniform region binning of NCC format Hi-C contact data and output as a sparse .npz format'

OUT_FORMATS = {'NPZ':'SciPy parse matrices stored as a NumPy archive (a .npz file with special keys)'}

CHR_KEY_SEP = ' '
DEFAULT_FORMAT = 'NPZ'
DEFAULT_BIN_SIZE = 50.0
DEFAULT_MIN_BINS = 2
M_SIZE = 500 # Size of submatrices in loading dict
PRINT_FREQ = 200000
MIN_TRANS_COUNT = 5

def bin_ncc(ncc_in, out_file=None, bin_size=DEFAULT_BIN_SIZE, format=DEFAULT_FORMAT,
            min_bins=DEFAULT_MIN_BINS, min_trans=MIN_TRANS_COUNT, dtype=np.uint8):

  from nuc_tools import util, io
  
  if not out_file:
    file_root, file_ext = os.path.splitext(ncc_in)
    
    if file_ext.lower() == '.gz':
      file_root, file_ext = os.path.splitext(file_root)
  
    out_file = '%s_%dk.%s' % (file_root, int(bin_size), format.lower())
  
  util.info('Reading %s' % ncc_in)
  
  bin_bp = int(bin_size * 1e3)
  contact_blocks = defaultdict(dict)
  counts = defaultdict(int)
  
  key_bin_size = M_SIZE * bin_bp
  n_contacts = 0
  exclude_contigs = set()
  min_bin = {}
  max_bin = defaultdict(int)
  get_min_bin = min_bin.get
  inf = float('inf')
  
  t0 = time.time()
  
  if format == 'NPZ':
    # Pre-read check to filter very small contigs
    
    with io.open_file(ncc_in, partial=True) as in_file_obj:
      for i, line in enumerate(in_file_obj):
        if (i == 1e5) and (len(max_bin) < 200):
          break
          
        chr_a, start_a, end_a, f_start_a, f_end_a, strand_a, \
          chr_b, start_b, end_b, f_start_b, f_end_b, strand_b, \
          ambig_group, pair_id, swap_pair = line.split()      

        pos_a = int(f_start_a) if strand_a == '+' else int(f_end_a)
        pos_b = int(f_start_b) if strand_b == '+' else int(f_end_b)
          
        bin_a = int(pos_a/bin_bp)
        bin_b = int(pos_b/bin_bp)

        if bin_a < get_min_bin(chr_a, inf):
          min_bin[chr_a] = bin_a

        if bin_b < get_min_bin(chr_b, inf):
          min_bin[chr_b] = bin_b              

        if bin_a > max_bin[chr_a]:
          max_bin[chr_a] = bin_a

        if bin_b > max_bin[chr_b]:
          max_bin[chr_b] = bin_b

        if i % PRINT_FREQ == 0:
          util.info("  Inspected {:,} lines, found {:,} chromosomes/contigs in {:.2f} s ".format(i, len(min_bin), time.time()-t0), line_return=True)
    
      else:
        util.info("  Inspected {:,} lines, found {:,} chromosomes/contigs in {:.2f} s ".format(i, len(min_bin), time.time()-t0), line_return=True)
        
        for chromo in min_bin:
          if (max_bin[chromo] - min_bin[chromo]) < min_bins:
            exclude_contigs.add(chromo)
          
        util.info('Excluding {:,} small contigs from {:,}'.format(len(exclude_contigs), len(min_bin)))
        min_bin = {}
        max_bin = defaultdict(int)
        get_min_bin = min_bin.get
        t0 = time.time() 
          
    with io.open_file(ncc_in) as in_file_obj:
      defer_ambig = []
      n_defer = 0
      
      for line in in_file_obj:
        chr_a, start_a, end_a, f_start_a, f_end_a, strand_a, \
          chr_b, start_b, end_b, f_start_b, f_end_b, strand_b, \
          ambig_group, pair_id, swap_pair = line.split()      
        
        if chr_a in exclude_contigs:
          continue
        
        if chr_b in exclude_contigs:
          continue
        
        n_contacts += 1

        pos_a = int(f_start_a) if strand_a == '+' else int(f_end_a)
        pos_b = int(f_start_b) if strand_b == '+' else int(f_end_b)
          
        bin_a = int(pos_a/bin_bp)
        bin_b = int(pos_b/bin_bp)
        
        if bin_a < get_min_bin(chr_a, inf):
          min_bin[chr_a] = bin_a

        if bin_b < get_min_bin(chr_b, inf):
          min_bin[chr_b] = bin_b              

        if bin_a > max_bin[chr_a]:
          max_bin[chr_a] = bin_a

        if bin_b > max_bin[chr_b]:
          max_bin[chr_b] = bin_b
        
        key_bin_a = int(pos_a/key_bin_size)
        key_bin_b = int(pos_b/key_bin_size)
         
        bin_a = bin_a % M_SIZE
        bin_b = bin_b % M_SIZE
        
        if chr_a > chr_b:
          chr_a, chr_b, bin_a, bin_b, key_bin_a, key_bin_b = chr_b, chr_a, bin_b, bin_a, key_bin_b, key_bin_a
        
        elif chr_a == chr_b and bin_b < bin_a:
          bin_a, bin_b, key_bin_a, key_bin_b = bin_b, bin_a, key_bin_b, key_bin_a
          
        ag, selected = ambig_group.split('.')

        if ag[0] != '1': # Ambiguous; multiple mapping pairs
        
          if ag[0] != '0': # First in group
            n_defer = int(ag[0])
            defer_ambig = [(selected, chr_a, chr_b, bin_a, bin_b, key_bin_a, key_bin_b),]
          
          else:
            defer_ambig.append((selected, chr_a, chr_b, bin_a, bin_b, key_bin_a, key_bin_b))
            
          if len(defer_ambig) == n_defer: # Last in group; group full
            pairs = [x[1:] for x in defer_ambig if x[0] != '0'] # Only active
            n_defer = 0
            
            if len(pairs) > 1:
              pairs.sort()
              
              while (len(pairs) > 1) and (pairs[0] == pairs[1]): # Ambig pairs are between same bins; allowed
                pairs = pairs[1:]
              
            if len(pairs) == 1:  
              chr_a, chr_b, bin_a, bin_b, key_bin_a, key_bin_b == pairs[0] # Resolved; same bins
              
            else: # Ambiguous (or completely inactive)
              continue

          else: # Do nothing until ambig group full
            continue
        
        elif selected == '0': # Unambigous but inactive
          n_defer = 0
          continue
        
        n_defer = 0
        chromo_key = (chr_a, chr_b)
        bin_key = (key_bin_a, key_bin_b)
        
        if bin_key in contact_blocks[chromo_key]:
          mat = contact_blocks[chromo_key][bin_key]
        else:
          if chr_a == chr_b:
            mat = np.zeros((M_SIZE, M_SIZE), dtype=dtype)
          else:
            mat = sparse.dok_matrix((M_SIZE, M_SIZE), dtype=dtype)
          
          contact_blocks[chromo_key][bin_key] = mat
         
          
        if n_contacts % PRINT_FREQ == 0:
          util.info("  Processed {:,} contacts for {:,} chromosomes/contigs in {:.2f} s ".format(n_contacts, len(min_bin), time.time()-t0), line_return=True)
        
        c = mat[bin_a, bin_b]
        
        if c > 254:
          current_dtype = mat.dtype
          
          if current_dtype == 'uint8':
            if hasattr(mat, 'toarray'):
              mat = mat.toarray()
          
            mat = contact_blocks[chromo_key][bin_key] = mat.astype(np.uint16)
          
          elif c > 65534 and current_dtype == 'uint16':
            mat = contact_blocks[chromo_key][bin_key] = mat.astype(np.uint32)
        
        mat[bin_a, bin_b] = c + 1
        counts[chromo_key] += 1
    
    util.info("  Processed {:,} contacts for {:,} chromosomes/contigs in {:.2f} s ".format(n_contacts, len(min_bin), time.time()-t0), line_return=True)
    
    util.info('Saving data')
    contacts = {}
    n_exc_trans = 0
    
    for chromo_key in sorted(counts):
      chr_a, chr_b = chromo_key
      min_a, max_a = min_bin[chr_a], max_bin[chr_a]
      min_b, max_b = min_bin[chr_b], max_bin[chr_b]
      min_a = min_b = 0
      a = 1 + max_a - min_a
      b = 1 + max_b - min_b
     
      if a < min_bins:
        del contact_blocks[chromo_key]
        continue

      if b < min_bins:
        del contact_blocks[chromo_key]
        continue      
      
      pair_dtype = dtype
      
      for mat in contact_blocks[chromo_key].values():
        if mat.dtype == 'uint16':
          pair_dtype = 'uint16'
          break
        
        elif mat.dtype == 'uint8':
          pair_dtype = 'uint8'
      
      mat = np.zeros((a, b), dtype=pair_dtype)
      
      for key_bin_a, key_bin_b in contact_blocks[chromo_key]:
        sub_mat = contact_blocks[chromo_key][(key_bin_a, key_bin_b)]
        
        if hasattr(sub_mat, 'toarray'):
          sub_mat = sub_mat.toarray()
        
        p1 = (key_bin_a * M_SIZE) - min_a
        p2 = min(a, p1 + M_SIZE)
        p1 = max(0, p1)
        
        q1 = (key_bin_b * M_SIZE) - min_b
        q2 = min(b, q1 + M_SIZE)
        q1 = max(0, q1)
        
        r1 = 0 if key_bin_a else min_a # Is offset in first bin
        r2 = 0 if key_bin_b else min_b
        
        #print chromo_key, key_bin_a, key_bin_b,  p1, p2, q1, q2, r1, r1+(p2-p1), r2, r2+(q2-q1)
        mat[p1:p2, q1:q2] = sub_mat[r1:r1+(p2-p1), r2:r2+(q2-q1)]
        
      del contact_blocks[chromo_key]
      
      if chr_a != chr_b and min_trans and  mat.sum() < min_trans:
        del mat
        n_exc_trans += 1
        continue
      
      key = chr_a + CHR_KEY_SEP + chr_b
      
      if chr_a == chr_b:
        contacts[key] = sparse.csr_matrix(mat)
      else:
        contacts[key] = sparse.coo_matrix(mat)
      
      del mat
      
      # Store bin offsets and spans
      contacts[chr_a] = np.array([min_a, a])
      contacts[chr_b] = np.array([min_b, b])
    
    contacts['params'] = np.array([bin_size, min_bins])  
     
    np.savez_compressed(out_file, **contacts)    
  
  if n_exc_trans:
    util.info('Excluded {:,} inter-chromosome pairs due to low contact counts (< {:,})'.format(n_exc_trans, min_trans))    
  
  util.info('Written {:,} contacts to {}'.format(n_contacts,out_file))
  
        
def main(argv=None):
  
  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  avail_fmts = ', '.join(['%s; %s' % x for x in OUT_FORMATS.items()])

  if argv is None:
    argv = sys.argv[1:]
  
  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'
  arg_parse = ArgumentParser(prog='nuc_tools ' + PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(nargs=1, metavar='NCC_FILE', dest='i',
                         help='Input NCC format file containing Hi-C contact data. May be Gzipped.')
 
  arg_parse.add_argument('-s', type=float, metavar='KB_BIN_SIZE',
                         default=DEFAULT_BIN_SIZE, dest='s',
                         help='Region bin size in kb, for grouping contacts')

  arg_parse.add_argument('-o', metavar='OUT_FILE', default=None, dest='o',
                         help='Optional output file name. If not specified, the output file ' \
                         'will be put in the same directory as the input and ' \
                         'automatically named according to the bin size and output format.')
  
  ## Could add different output formats in future                        
  #arg_parse.add_argument('-f', metavar='OUT_FORMAT', default=DEFAULT_FORMAT, dest='f',
  #                       help='Output file format. Default: %s. Available: %s' % (DEFAULT_FORMAT, avail_fmts))

  arg_parse.add_argument('-m', '--min-bin-count', default=DEFAULT_MIN_BINS, metavar='MIN_BINS', type=int, dest='m',
                         help='The minimum number of bins for chromosomes/contigs; those with fewer ' \
                              'than this are excluded from output. Default: %d' % DEFAULT_MIN_BINS)

  arg_parse.add_argument('-t', '--min-trans-count', default=MIN_TRANS_COUNT, metavar='MIN_TRANS_COUNT', type=int, dest='t',
                         help='The minimum number contacts for inter-chromosomal contact matrices; ' \
                              'those with fewer than this are excluded from output. Default: %d' % MIN_TRANS_COUNT)

  args = vars(arg_parse.parse_args(argv))

  in_file  = args['i'][0]
  out_file = args['o']
  bin_size = args['s']
  fmt   = DEFAULT_FORMAT # args['f'].upper()  
  min_bins = args['m']
  min_trans = args['t']
  
  invalid_msg = io.check_invalid_file(in_file)
  
  if invalid_msg:
    util.critical(invalid_msg)
  
  if out_file and io.is_same_file(in_file, out_file):
    util.critical('Input file cannot be the same as the output file')

  if fmt not in OUT_FORMATS:
    msg = 'Output file format "%s" not known. Available: %s.' % (fmt, ', '.join(sorted(OUT_FORMATS)))
    util.critical(msg)  
    
  bin_ncc(in_file, out_file, bin_size, fmt, min_bins, min_trans)
 

if __name__ == '__main__':
  
  from time import time
  
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  
  main()
  
