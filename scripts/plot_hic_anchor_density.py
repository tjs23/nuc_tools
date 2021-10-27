import sys, os
import numpy as np
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict
from nuc_tools import util, io
from formats import bed, gff, ncc

def _load_data_points(data_files):

  track_data_dicts = {}
  file_colors = {}
  out_styles = {}
  
  for label, colors, file_path, file_format, feature, in data_files:
    util.info('Loading {}'.format(file_path))
    file_colors[label] = colors
  
    if file_format == 'BED':
      track_data_dicts[label] = bed.load_data_track(file_path)
 
    elif file_format == 'GFF':
      track_data_dicts[label] = list(gff.load_data_track(file_path, [feature]).values())[0]
  
  out_dicts = {}

  for label in track_data_dicts:
    data_dict = track_data_dicts[label]
    colors = file_colors[label]
  
    if len(colors) == 1:
      key = label+'_mid'
 
      out_dicts[key] = {}
      out_styles[key] = (colors[0], '-')

      for chromo, data_list in data_dict.items():
        p1 = data_list['pos1'].astype(int)
        p2 = data_list['pos2'].astype(int)
        s_neg = data_list['strand'] == 0
 
        mids = ((p1 + p2)//2).astype(int)
        mids[s_neg] *= -1
        out_dicts[key][chromo] = np.sort(mids)
    
    else:
      key1 = label+'_start'
      key2 = label+'_end'
      out_dicts[key1] = {}
      out_dicts[key2]  = {}
      out_styles[key1] = (colors[0], '-')
      out_styles[key2] = (colors[1], '-')

      for chromo, data_list in data_dict.items():
        p1 = data_list['pos1'].astype(int)
        p2 = data_list['pos2'].astype(int)
        s_neg = data_list['strand'] == 0
 
        starts = p1.copy()
        starts[s_neg] = p2[s_neg]
        starts_order = starts.argsort()
        starts[s_neg] *= -1
 
        ends = p2.copy()
        ends[s_neg] = p1[s_neg]
        ends_order = ends.argsort()
        ends[s_neg] *= -1
 
        out_dicts[key1][chromo] = starts[starts_order]
        out_dicts[key2][chromo] = ends[ends_order]

  
  return out_dicts, out_styles

from numba import njit, int64, float64, int32, prange

@njit(int64[:](int64[:], int64[:], int64[:], int64, int64))
def _add_to_map(hic_pos, data_map, anchor_points, n_bins, bin_size):
  
  n_hic = len(hic_pos)
  n_points = len(anchor_points)
  lim = n_bins * bin_size
  j0 = 0
  
  
  
  for i in range(n_hic): # Ordered
    p = hic_pos[i]
    
    for j in range(j0, n_points): # Ordered
      a_pos = anchor_points[j]
      is_neg = a_pos < 0
      a_pos = abs(a_pos)
      a1 = a_pos-lim
      a2 = a_pos+lim
      if a2 < p:
        j0 = j+1
      
      if  a1 < p < a2:
        delta_bins = (p-a_pos) // bin_size # Negative if hi-c point is before anchor
        
        if is_neg:
          k = n_bins - delta_bins - 1  # Zero/middle at n_bins
        else:
          k = n_bins + delta_bins  # Zero/middle at n_bins
          
        data_map[k] += 1
        
      elif a1 > p:
        break
      
      
      
  return data_map
  
  
def plot_hic_anchor_density(ncc_path, anchor_dicts, anchor_styles, max_sep=5000, bin_size=20):
  
  contact_data = defaultdict(list)
  
  with io.open_file(ncc_path) as file_obj:
   
    util.info('Reading {}'.format(ncc_path))
    n = 0
    
    for line in file_obj:
       chr_a, f_start_a, f_end_a, start_a, end_a, strand_a, \
       chr_b, f_start_b, f_end_b, start_b, end_b, strand_b, \
       ambig_code, pair_id, swap_pair = line.split()
       """
       if strand_a == '+':
         contact_data[chr_a].append((start_a, f_end_a)
       else:
         contact_data[chr_a].append((f_start_a, end_b))
      
       if strand_b == '+':
         contact_data[chr_b].append((start_b, f_end_b)
       else:
         contact_data[chr_b].append((f_start_b, end_b))
       """
       pos_a = int(f_end_a if strand_a == '+' else f_start_a)
       pos_b = int(f_end_b if strand_b == '+' else f_start_b)
     
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
  n_chromo = len(contact_data)
  anchor_count = defaultdict(int)
  
  for c, chromo in enumerate(contact_data):
    hic_pos = np.array(contact_data[chromo])
    hic_pos = hic_pos[hic_pos.argsort()]
    
    if c and c % 1000 == 0:
      util.info(f' .. {c:,} of {n_chromo:,}', line_return=True)
    
    for label, data_dict in anchor_dicts.items():
      if chromo not in data_dict:
        continue
      
      anchor_points = data_dict[chromo]
      anchor_points = anchor_points[np.abs(anchor_points).argsort()]
      anchor_count[label] += len(anchor_points)
      data_map = anchor_maps[label]
      
      _add_to_map(hic_pos, data_map, anchor_points, n_bins, bin_size)
  util.info(f' .. {c:,} of {n_chromo:,}', line_return=True)
  
  fig, ax = plt.subplots()
  fig.set_size_inches(8.0, 8.0)
       
  ax.set_title('Anchor Hi-C distribution')
  ax.set_xlabel('Separation from anchor (bp)')
  ax.set_ylabel(f'Mean Hi-C contact count ({bin_size} bp bins)')
  ax.set_xlim((-max_sep, max_sep))
  ax.set_ylim((0.8, 2.2))
  
  x_vals = np.linspace(-max_sep, max_sep, (2*max_sep)//bin_size)
  
  for label, anchor_map in anchor_maps.items():
    color, linestyle = anchor_styles[label] 
    anchor_map = anchor_map.astype(float)
    anchor_map /= anchor_count[label]
    
    ax.plot(x_vals, anchor_map, label=label, color=color, linestyle=linestyle, alpha=0.3)
    
  #save_path = '/data/dino_hi-c/HiC_anchor_sep_distrib_' + label + '.pdf'
  #plt.savefig(save_path, dpi=300)
  
  ax.axvline(0.0, linestyle='--', linewidth=0.5, alpha=0.5, color='#808080')
  ax.legend(fontsize=8)
  
  plt.show()
  
      
      

if __name__ == '__main__':
   
  data_files = (('DVNP',               ['#505050',], '/data/dino_hi-c/ChIP-seq_peaks/DVNP-HiC2_peaks.narrowPeak', 'BED', None),
                ('H2A',                ['#008000',], '/data/dino_hi-c/ChIP-seq_peaks/H2A-HiC2_peaks.narrowPeak', 'BED', None),
                #('Trin_gene', ['#FF2000','#0080FF',], '/data/dino_hi-c/hem_flye_4_ssRNA_Trinity.gff3', 'GFF', 'gene'),
                ('Trin_exon', ['#FF2000','#0080FF'], '/data/dino_hi-c/hem_flye_4_ssRNA_Trinity.gff3', 'GFF', 'exon')
                )
  anchor_dicts, anchor_styles = _load_data_points(data_files)
   
  ncc_path = '/data/dino_hi-c/Hi-C_contact_data/SLX-17943_HEM_FLYE_4_Hi-C.ncc.gz'
 
  plot_hic_anchor_density(ncc_path, anchor_dicts, anchor_styles, max_sep=4000, bin_size=50)
