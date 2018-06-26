ncc_path = '/data/hi-c/pop_HybridES0418_unambig.ncc'
ncc_path = '/data/hi-c/pop_HybridES0418.ncc.gz'

import numpy as np

from collections import defaultdict

from nuc_tools import util, io

from formats import fasta

from matplotlib import pyplot as plt

chromos = ('chr1', 'chr2', 'chr3', 'chr4', 'chrX')
n = len(chromos) * 2

n_cols = 2
n_rows = n/n_cols

val_max = 2.1e8
val_range = (0, val_max)
bin_size = 1e4

n_bins = int(val_max/bin_size)

cis_counts = {}
homolog_counts = {}
seq_seps = {}
seq_seqs_counts = {}
ag_sizes = defaultdict(int)

for chromo in chromos:
  chr_a = chromo + '.a'
  chr_b = chromo + '.b'
  
  cis_counts[chr_a] = np.zeros(n_bins, float)
  homolog_counts[chr_a] = np.zeros(n_bins, float)
  seq_seps[chr_a] = np.zeros(n_bins, float)
  seq_seqs_counts[chr_a] = np.zeros(n_bins, float)

  cis_counts[chr_b] = np.zeros(n_bins, float)
  homolog_counts[chr_b] = np.zeros(n_bins, float)
  seq_seps[chr_b] = np.zeros(n_bins, float)
  seq_seqs_counts[chr_b] = np.zeros(n_bins, float)

hom_ags = set()

with io.open_file(ncc_path) as file_obj:
    
  for line in file_obj:
  
    chr_a, start_a, end_a, f_start_a, f_end_a, strand_a, \
      chr_b, start_b, end_b, f_start_b, f_end_b, strand_b, \
      ambig_group, pair_id, swap_pair = line.split()
    
    root_a = chr_a.split('.')[0]
    root_b = chr_b.split('.')[0]
    
    if chr_a not in cis_counts:
      continue
    if chr_b not in cis_counts:
      continue
    
    """  
    if strand_a == '+':
      pos_a = int(f_start_a)
    else:
      pos_a = int(f_end_a)

    if strand_b == '+':
      pos_b = int(f_start_b)
    else:
      pos_b = int(f_end_b)
    """
    #delta = abs(pos_b-pos_a)
    #delta = np.log10(delta or 1.0)
    
    #bin1 = int(pos_a/bin_size)
    #bin2 = int(pos_b/bin_size)
    
    if chr_a == chr_b:
      pass
      #cis_counts[chr_a][bin1] += 1
      #cis_counts[chr_a][bin2] += 1
  
    elif root_a == root_b: # Homologue trans
      #homolog_counts[chr_a][bin1] += 1
      #homolog_counts[chr_b][bin2] += 1
      #seq_seps[chr_a][bin1] += delta
      #seq_seps[chr_b][bin2] += delta
      #seq_seqs_counts[chr_a][bin1] += 1
      #seq_seqs_counts[chr_b][bin2] += 1
      hom_ags.add(ambig_group)
    
    ag_sizes[ambig_group] += 1

sizes = defaultdict(int)
for ag in hom_ags:
   sizes[ag_sizes[ag]] += 1


print sizes # {1: 15,960, 2: 6,777,487, 4: 2,751,994}

sys.exit()

 
fig, axarr = plt.subplots(n_rows, n_cols, sharex=False)
 
for i, chromo in enumerate(sorted(cis_counts)):
  row = int(i // n_cols)
  col = i % n_cols
  
  if n_cols == 1:
    ax = axarr[row]
  else:
    ax = axarr[row, col]
  
  ax.set_ylabel(chromo)
  
  cisc = cis_counts[chromo]
  
  nz = seq_seqs_counts[chromo].nonzero()
  ssep = np.zeros(n_bins, float)
  ssep[nz] = (seq_seps[chromo][nz]/seq_seqs_counts[chromo][nz])
  homc = homolog_counts[chromo]
  
  cisc /= cisc.sum()
  homc /= homc.sum()
  
  nz = (cisc * homc).nonzero()
  entropy = homc[nz] * np.log2(homc[nz]/cisc[nz])
  #entropy[nz] = homc[nz]/cisc[nz]
  
  m = entropy.max()
  ax.set_ylim((entropy.min() - 0.05 * m , m * 1.05))
  m = ssep.max()
  ax.set_xlim((ssep.min() - 0.05 * m, m * 1.05))
  
  ax.scatter(ssep[nz], entropy, alpha=0.25, color='#0080FF')
  
  #cisc += cisc.mean()
  #homc += homc.mean()
  #x_vals = np.arange(0, val_max, bin_size)
  #ax.plot(x_vals, homc/cisc, label='Cis', alpha=0.5, color='#FF4000')
  
  if row == n_rows-1:
    ax.set_xlabel('Seq. sep')
  
  #ax.legend()

plt.show()

# Work at the restriction fragment level? - Though reads are much smaller than this, this is the precision of any structural exclusion.

# 

