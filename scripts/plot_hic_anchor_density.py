import sys, os
import numpy as np
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict
from nuc_tools import util, io
from formats import bed, gff, ncc

def _load_data_points(data_files):

  track_data_dicts = {}
  use_middle = {}
  
  for label, file_path, file_format, feature, is_middle in data_files:
    util.info('Loading {}'.format(file_path))
    use_middle[label] = is_middle
  
    if file_format == 'BED':
      track_data_dicts[label] = bed.load_data_track(file_path)
 
    elif file_format == 'GFF':
      track_data_dicts[label] = list(gff.load_data_track(file_path, [feature]).values())[0]
  
  out_dicts = {}

  for label in track_data_dicts:
    data_dict = track_data_dicts[label]
    
    if use_middle[label]:
      out_dicts[label+'_mid'] = {}

      for chromo, data_list in data_dict.items():
        p1 = data_list['pos1'].astype(int)
        p2 = data_list['pos2'].astype(int)
        s_neg = data_list['strand'] == 0
 
        mids = ((p1 + p2)//2).astype(int)
        mids[s_neg] *= -1 
        out_dicts[label+'_mid'][chromo] = np.sort(mids)
    
    else:
      out_dicts[label+'_start'] = {}
      out_dicts[label+'_end']  = {}

      for chromo, data_list in data_dict.items():
        p1 = data_list['pos1'].astype(int)
        p2 = data_list['pos2'].astype(int)
        s_neg = data_list['strand'] == 0
 
        starts = p1[:]
        starts[s_neg] = p2[s_neg]
        starts[s_neg] *= -1
 
        ends = p2[:]
        ends[s_neg] = p1[s_neg]
        ends[s_neg] *= -1
 
        out_dicts[label+'_start'][chromo] = np.sort(starts)
        out_dicts[label+'_end'][chromo] = np.sort(ends)

  
  return out_dicts

from numba import njit, int64, float64, int32, prange

@njit(int64[:](int64[:], int64[:], int64[:], int64, int64))
def _add_to_map(hic_pos, data_map, anchor_points, n_bins, bin_size):
  
  n_hic = len(hic_pos)
  n_points = len(anchor_points)
  lim = n_bins * bin_size
  
  for i in range(n_hic):
    p = hic_pos[i]
    
    for j in range(n_points): # Ordered
      a_pos = anchor_points[j]
      a1 = a_pos-lim
      a2 = a_pos+lim
      
      if  a1 < p < a2:
        
        k = ((a_pos - p) // bin_size) + n_bins # Zero at n_bins
        data_map[k] += 1
        
      elif a1 > p:
        break
  
      
  return data_map
  
  
def plot_hic_anchor_density(ncc_path, anchor_dicts, max_sep=2000, bin_size=10):
  
  contact_data = defaultdict(list)
  
  with io.open_file(ncc_path) as file_obj:
   
    util.info('Reading {}'.format(ncc_path))
    n = 0
    
    for line in file_obj:
       chr_a, f_start_a, f_end_a, start_a, end_a, strand_a, \
       chr_b, f_start_b, f_end_b, start_b, end_b, strand_b, \
       ambig_code, pair_id, swap_pair = line.split()
       
       pos_a = int(f_start_a if strand_a == '+' else f_end_a)
       pos_b = int(f_start_b if strand_b == '+' else f_end_b)
     
       contact_data[chr_a].append(pos_a)
       contact_data[chr_b].append(pos_b)
       
       n += 1
       if n % 100000 == 0:
         util.info(' .. found {:,} contacts'.format(n), line_return =True)
         
         #if n > 1e7:
         #  break
   
  #util.info('Binning contacts')
  anchor_maps = {}
  
  for label, data_dict in anchor_dicts.items():
    anchor_maps[label] = np.zeros((max_sep // bin_size)*2, int)
    
  util.info('Aggregating maps')
  n_bins = max_sep // bin_size
  
  for chromo in contact_data:
    hic_pos = np.array(contact_data[chromo])
    
    for label, data_dict in anchor_dicts.items():
      if chromo not in data_dict:
        continue
      
      anchor_points = data_dict[chromo]
      data_map = anchor_maps[label]
    
      _add_to_map(hic_pos, data_map, anchor_points, n_bins, bin_size)
  
  fig, ax = plt.subplots()
  fig.set_size_inches(8.0, 8.0)
       
  ax.set_title('Anchor: ' + label)
  ax.set_xlabel('Separation from anchor (kb)')
  ax.set_ylabel('Count')
  
  for label, anchor_map in anchor_maps.items():
     
    anchor_map = anchor_map.astype(float)
  
    
    ax.plot(anchor_map, label=label, alpha=0.5)
    
  #save_path = '/data/dino_hi-c/HiC_anchor_sep_distrib_' + label + '.pdf'
  #plt.savefig(save_path, dpi=300)
  
  ax.legend(fontsize=8)
  
  plt.show()
  
      
      

if __name__ == '__main__':
   
  data_files = (('DVNP', '/data/dino_hi-c/ChIP-seq_peaks/DVNP-HiC2_peaks.narrowPeak', 'BED', None, True),
                ('H2A', '/data/dino_hi-c/ChIP-seq_peaks/H2A-HiC2_peaks.narrowPeak', 'BED', None, True),
                ('Trinity_ssRNA_gene', '/data/dino_hi-c/hem_flye_4_ssRNA_Trinity.gff3', 'GFF', 'gene', False),
                ('Trinity_ssRNA_exon', '/data/dino_hi-c/hem_flye_4_ssRNA_Trinity.gff3', 'GFF', 'exon', True),
                ('Trinity_ssRNA_exon', '/data/dino_hi-c/hem_flye_4_ssRNA_Trinity.gff3', 'GFF', 'exon', False)
                )
  anchor_dicts = _load_data_points(data_files)
  
  ncc_path = '/data/dino_hi-c/Hi-C_contact_data/SLX-17943_HEM_FLYE_4_Hi-C.ncc.gz'
 
  plot_hic_anchor_density(ncc_path, anchor_dicts, max_sep=2000, bin_size=10)
