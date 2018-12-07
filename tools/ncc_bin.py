import sys, os, math, time
import numpy as np

from collections import defaultdict
from scipy import sparse
from sys import stdout

PROG_NAME = 'ncc_bin'
VERSION = '1.0.0'
DESCRIPTION = 'Perform uniform region binning of NCC format Hi-C contact data and output in various formats'

OUT_FORMATS = {'NPZ':'SciPy parse matrices stored as a NumPy archive (a .npz file with special keys)'}

CHR_KEY_SEP = ' '
DEFAULT_FORMAT = 'NPZ'
DEFAULT_BIN_SIZE = 50.0
DEFAULT_MIN_BINS = 2

def bin_ncc(ncc_in, out_file=None, bin_size=DEFAULT_BIN_SIZE, format=DEFAULT_FORMAT,
            min_bins=DEFAULT_MIN_BINS, dtype=np.uint8):

  from nuc_tools import util, io
  
  if not out_file:
    file_root, file_ext = os.path.splitext(ncc_in)
    
    if file_ext.lower() == '.gz':
      file_root, file_ext = os.path.splitext(file_root)
  
    out_file = '%s_%dk.%s' % (file_root, int(bin_size), format.lower())
  
  util.info('Reading %s' % ncc_in)
  
  bin_bp = int(bin_size * 1e3)
  contacts = {}
  counts = defaultdict(int)
  
  bin_step = 100
  n_contacts = 0
  min_bin = {}
  max_bin = defaultdict(int)
  get_min_bin = min_bin.get
  inf = float('inf')
  
  t0 = time.time()
  
  if format == 'NPZ':
    with io.open_file(ncc_in) as in_file_obj:
 
      for line in in_file_obj:
        n_contacts += 1
        
        chr_a, start_a, end_a, f_start_a, f_end_a, strand_a, \
          chr_b, start_b, end_b, f_start_b, f_end_b, strand_b, \
          ambig_group, pair_id, swap_pair = line.split()      

        if strand_a == '+':
          pos_a = int(f_start_a)
        else:
          pos_a = int(f_end_a)

        if strand_b == '+':
          pos_b = int(f_start_b)
        else:
          pos_b = int(f_end_b)
          
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
        
        if chr_a > chr_b:
          chr_a, chr_b, bin_a, bin_b = chr_b, chr_a, bin_b, bin_a
        
        elif chr_a == chr_b and bin_b < bin_a:
          bin_a, bin_b = bin_b, bin_a
        
        key = chr_a + CHR_KEY_SEP +  chr_b
        
        if key in contacts:
          a, b = contacts[key].shape
          p, q = a, b
          
          if (a <= bin_a) or (b <= bin_b):
            while a <= bin_a:
              a += bin_step
 
            while b <= bin_b:
              b += bin_step
            
            contacts[key] = np.pad(contacts[key], [(0,a-p), (0,b-q)], 'constant')
         
        else:
          a, b = bin_step, bin_step
          
          while a <= bin_a:
            a += bin_step
          
          while b <= bin_b:
            b += bin_step
          
          contacts[key] = np.zeros((a, b), dtype=dtype)
          
        if n_contacts % 100000 == 0:
          stdout.write("\r  Processed {:,} contacts in {:.2f} s ".format(n_contacts, time.time()-t0))
          stdout.flush()
        
        c = contacts[key][bin_a, bin_b]
        
        if c > 254:
          current_dtype = contacts[key].dtype
          
          if current_dtype == 'uint8':
            contacts[key] = contacts[key].astype(np.uint16)
          
          elif c > 65534 and current_dtype == 'uint16':
            contacts[key] = contacts[key].astype(np.uint32)
        
        contacts[key][bin_a, bin_b] = c + 1
        counts[key] += 1
    
    stdout.write("\r  Processed {:,} contacts in {:.2f} s\n".format(n_contacts, time.time()-t0))
    stdout.flush()
    
    util.info('Saving data')
 
    for key in sorted(counts):
      chr_a, chr_b = key.split(CHR_KEY_SEP)
      min_a, max_a = min_bin[chr_a], max_bin[chr_a]
      min_b, max_b = min_bin[chr_b], max_bin[chr_b]
      
      if max_a - min_a < min_bins:
        del contacts[key]
        continue

      if max_b - min_b < min_bins:
        del contacts[key]
        continue      
      
      if chr_a == chr_b:
        min_a = min_b = min(min_a, min_b) # Cis should always be square
        max_a = max_b = max(max_a, max_b)
        contacts[key] = sparse.csr_matrix(contacts[key][min_a:max_a+1,min_b:max_b+1])
      else:
        contacts[key] = sparse.coo_matrix(contacts[key][min_a:max_a+1,min_b:max_b+1])
      
      # Store bin offsets and spans
      contacts[chr_a] = np.array([min_a, max_a-min_a+1])
      contacts[chr_b] = np.array([min_b, max_b-min_b+1])
    
    contacts['params'] = np.array([bin_size, min_bins])  
     
    np.savez_compressed(out_file, **contacts)    
  
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
                         help='Optional output file name. If not specified the output file' \
                         'will be put in the same directory as the input and' \
                         'automatically named according to the bin size and output format')
                         
  arg_parse.add_argument('-f', metavar='OUT_FORMAT', default=DEFAULT_FORMAT, dest='f',
                         help='Output file format. Default: %s. Available: %s' % (DEFAULT_FORMAT, avail_fmts))

  arg_parse.add_argument('-m', default=DEFAULT_MIN_BINS, metavar='MIN_BINS', type=int, dest='m',
                         help='The minimum number of bins for chromosomes/contigs; those with fewer than this are excluded from output')

  args = vars(arg_parse.parse_args(argv))

  in_file  = args['i'][0]
  out_file = args['o']
  bin_size = args['s']
  format   = args['f'].upper()  
  min_bins = args['m']
  
  invalid_msg = io.check_invalid_file(in_file)
  
  if invalid_msg:
    util.critical(invalid_msg)
  
  if out_file and io.is_same_file(in_file, out_file):
    util.critical('Input file cannot be the same as the output file')

  if format not in OUT_FORMATS:
    msg = 'Output file format "%s" not known. Available: %s.' % (format, ', '.join(sorted(OUT_FORMATS)))
    util.critical(msg)  
    
  bin_ncc(in_file, out_file, bin_size, format, min_bins)
 

if __name__ == '__main__':

  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
  main()
