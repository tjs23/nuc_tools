import sys,  os, math
import numpy as np


PROG_NAME = 'sequence_properties'
VERSION = '1.0.0'
DESCRIPTION = 'Creates data tracks describing derived DNA sequence properties (e.g GC vs AT ratio) given input sequences in FASTA format'
DEFAULT_BIN_SIZE = 1000
MIN_KNOWN = 20
REP_WINDOW = 10
DEFAULT_SMALLEST_CONTIG = 100

def sequence_properties(fastq_paths, out_path_root=None, bin_size=DEFAULT_BIN_SIZE, min_seq_len=DEFAULT_SMALLEST_CONTIG*1000):
  
  from formats import fasta, bed
  from nuc_tools import util, io
    
  if out_path_root:
    out_path_root = os.path.splitext(out_path_root)[0]
  else:
    out_path_root = os.path.splitext(fastq_paths[0])[0]
  
  gcat_data = {}
  nn_data = {}
  rep_data = {}
  total = 0
  
  r_idx = {r:i for i, r in enumerate('GCATN')}
  
  rep_step = int(REP_WINDOW/2)
  rep_frac = 1.0/REP_WINDOW
  probs_init = np.zeros(5, float)
  prior = np.full(5, 0.2)
  
  G = ord('G')
  C = ord('C')
  A = ord('A')
  T = ord('T')
  N = ord('N')
  
  from scipy.stats import entropy
  
  unique = np.unique
  ones = np.ones
  column_stack = np.column_stack
  
  for fastq_path in fastq_paths:
    util.info('Reading {}'.format(fastq_path))
    seq_dict = fasta.read_fasta(fastq_path, as_array=True)
  
    for seq_name, seq in seq_dict.items():
      n = len(seq)
      if n < min_seq_len:
        continue
      
      total += n
      util.info(' .. analysing {} over {:,} bp'.format(seq_name, n), line_return=True)
      n_bins = int(math.ceil(n/float(bin_size)))

      gcat_hist = np.zeros(n_bins)
      nn_hist = np.zeros(n_bins)
      rep_hist = np.zeros(n_bins)
  
      for i in range(n_bins):
        p1 = i*bin_size
        p2 = min(p1 + bin_size, n)
        sub_seq = seq[p1:p2]
        
        if len(sub_seq) < MIN_KNOWN:
          continue
        
        uniq, counts = unique(sub_seq, return_counts=True)
        count_dict = dict(zip(uniq, counts))
        
        nn = count_dict.get(N, 0)
        nn_hist[i] = nn/float(bin_size)
        
        triples = column_stack([sub_seq[:-2], sub_seq[1:-1], sub_seq[2:]])
        triples = triples[(triples != [N,N,N])[:,0]]
         
        if len(triples) < 1:
          continue
       
        uniq, counts = unique(triples, axis=0, return_counts=True)

        rep_hist[i] = entropy(counts, ones(len(counts)))
        
        if bin_size-nn < MIN_KNOWN:
          continue
        
        gc = count_dict.get(G, 0) + count_dict.get(C, 0)
        tt = bin_size-nn
        if gc and tt:
          ratio = gc/float(tt)
        elif gc:
          ratio = 0.0
            
        gcat_hist[i] = ratio
      
      gcat_data[seq_name] = util.hist_to_data_track(gcat_hist, bin_size)
      nn_data[seq_name] = util.hist_to_data_track(nn_hist, bin_size)
      rep_data[seq_name] = util.hist_to_data_track(rep_hist, bin_size)
      
  n_gcat = sum([len(gcat_data[c]) for c in gcat_data])
  
  gcat_file_path = out_path_root + '_fracGC.bed'
  
  util.info('Saving {:,} values for {:,} chromosomes/sequences totalling {:,} bp'.format(n_gcat, len(gcat_data), total))
  
  bed.save_data_track(gcat_file_path, gcat_data, as_float=True)
  
  util.info('Written {}'.format(gcat_file_path))
  
  n_nn = sum([len(nn_data[c]) for c in nn_data])

  nn_file_path = out_path_root + '_fracN.bed'

  util.info('Saving {:,} values for {:,} chromosomes/sequences totalling {:,} bp'.format(n_nn, len(nn_data), total))
  
  bed.save_data_track(nn_file_path, nn_data, as_float=True)
  
  util.info('Written {}'.format(nn_file_path))
  n_rep = sum([len(rep_data[c]) for c in rep_data])

  rep_file_path = out_path_root + '_trip_entropy.bed'

  util.info('Saving {:,} values for {:,} chromosomes/sequences totalling {:,} bp'.format(n_rep, len(rep_data), total))
  
  bed.save_data_track(rep_file_path, rep_data, as_float=True)
  
  util.info('Written {}'.format(rep_file_path))

    
def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)

  arg_parse.add_argument(metavar='FASTA_FILES', nargs='+', dest='i',
                         help='One or more input sequence files in FASTA format. e.g. of a genome. Accepts wildcards.')

  arg_parse.add_argument('-s', '--bin-size', default=DEFAULT_BIN_SIZE, metavar='BIN_SIZE', type=int, dest="s",
                         help='The binned region size for output data tracks, in base pairs. Default is %d bp.' % DEFAULT_BIN_SIZE)

  arg_parse.add_argument('-o', '--out-file-root',metavar='OUT_FILE_ROOT', dest='o',
                         help='Optional file path to specify where output data track files will be saved.' \
                              'All output file names will be appended with a label indicating the data track type.')

  arg_parse.add_argument('-m', default=DEFAULT_SMALLEST_CONTIG, metavar='MIN_CONTIG_SIZE', type=int,
                         help='The minimum chromosome/contig sequence length in kilobases for inclusion. Default is {}.'.format(DEFAULT_SMALLEST_CONTIG))
 
 
  args = vars(arg_parse.parse_args(argv))

  fasta_paths = args['i']
  out_path_root = args['o']
  bin_size = args['s']
  min_seq_len = args['m'] * 1000
  
  for fasta_path in fasta_paths:
    io.check_invalid_file(fasta_path)  
  
  sequence_properties(fasta_paths, out_path_root, bin_size, min_seq_len)
  

if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()

