import sys, os, time
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nuc_tools import io, util
from formats import sam, bed, gff
from collections import defaultdict

""" - Plot size distribs at anchors:  - as 2D frag length and anchor sep
      # For MNase 0.6 and 1.5 (comb)
      # Skip v. small contigs
      # Load anchor positions, per contig
      # For each MNase read, get contig, get size
      #   For each anchor on same contig
      #     Store size, anchor sep (closest point; min of ends), spearated by strand
"""
 
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
        out_dicts[label+'_mid'][chromo] = mids
    
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
 
        out_dicts[label+'_start'][chromo] = starts
        out_dicts[label+'_end'][chromo] = ends

  
  return out_dicts


from numba import njit, int64, float64, int32, prange

@njit(int64[:,:](int64[:,:], int64[:,:], int64[:], int64, int64, int64, int64))
def _add_to_map(regions, data_map, data_points, mid_col, n, min_size, bin_size):
  lim = mid_col-1
  n_regions = len(regions)
  n_points = len(data_points)
  p_start = 0
  
  for r in range(n_regions):
    p1 = regions[r,0]
    p2 = regions[r,1]
    row = int((p2-p1-min_size)//bin_size) # Sequence separation
    
    for p in range(p_start, n_points):
      a0 = data_points[p]

      if a0 < 0: # Neg strand
        a0 = abs(a0)
        a1 = a0-lim
        a2 = a0+lim
        if a1 < p1 < a2:
          j = a2-p2
 
          if p2 < a2:
            k = a2-p1
          else:
            k = n
 
        elif a1 < p2 < a2:
          j = 0
          k = a2-p1
 
        elif a1 > p2: # Points are sorted
          break
 
        else:
          continue
 
      else:
        a1 = a0-lim
        a2 = a0+lim
        if a1 < p1 < a2:
          j = p1-a1
 
          if p2 < a2:
            k = p2-a1
          else:
            k = n
 
        elif a1 < p2 < a2:
          j = 0
          k = p2-a1
 
        elif a1 > p2: # Points are sorted
          break
 
        else:
          continue
          
      if a2 < p1:
        p_start = p
 
      for i in range(j,k):
        data_map[row,i] += 1
      
  return data_map
  

def plot_bam_anchor_sep_size(bam_path, anchor_dicts, min_size=130, max_size=1000, max_sep=2000, bin_size=5):
  
  nx = 2*max_sep + 1
  ny = int((max_size-min_size)//bin_size)+bin_size
  mid_col = max_sep + 1
  
  anchor_maps = {label:np.zeros((ny, nx), int) for label in anchor_dicts}
  
  t0 = time.time()
  cmap = util.string_to_colormap('#FFFFFF,#0080FF,#FF0000,#000000')
  
  chromo_regions = defaultdict(list)
    
  chromo_sizes = dict(sam.get_bam_chromo_sizes(bam_path))

  for i, (rname, sam_flag, chromo, pos, mapq, cigar, mate_chromo, mate_pos, t_len, seq, qual) in enumerate(sam.bam_iterator(bam_path)):
    
    if i % 100000 == 0:
      util.info('  .. {:,} {:7.2f}s'.format(i, time.time()-t0), line_return=True)
    
    #if i > 1e6:
    #  break
    
    if chromo == '*':
      continue
    
    if mate_chromo != '=':
      continue
    
    if chromo_sizes[chromo] < max_sep:
      continue
    
    sam_flag = int(sam_flag)
     
    if sam_flag & 0x4: # R1 unmapped
      continue

    if sam_flag & 0x8: # R2 unmapped
      continue

    if sam_flag & 0x100: # Secondary
      continue
    
    p1 = int(pos)
    p2 = int(mate_pos)
    
    if p1 > p2:
      p1, p2, = p2, p1
    
    size = int(t_len) # len(seq) + p2 - p1
    p2 = p1 + size

    if size >= max_size:
      continue
    
    if size < min_size:
      continue
    
    chromo_regions[chromo].append((p1, p2))
    
    
    #s_neg = int(sam_flag) & 0x10
   
  util.info('  .. {:,} {:7.2f}s'.format(i, time.time()-t0), line_return=True)
  util.info('Aggregating maps')
  
  for chromo in chromo_regions:
    regions = np.array(sorted(chromo_regions[chromo]))
     
    for label, data_dict in anchor_dicts.items():
      if chromo not in data_dict:
        continue
      
      data_points = data_dict[chromo]
      data_points = data_points[np.abs(data_points).argsort()]
      data_map = anchor_maps[label]
      _add_to_map(regions, data_map, data_points, mid_col, nx, min_size, bin_size)  
 
  util.info(' Time taken {:7.2f}s'.format(time.time()-t0))
           
  
  xlabel_pos = np.arange(0, 2*max_sep+1, 500)
  xlabels = ['%.1f' % ((x-max_sep)*1e-3) for x in xlabel_pos]
  
  ystep = 100//bin_size
  y0  = ystep-divmod(min_size//bin_size, ystep)[1] # Offset to the next 100  
  ylabel_pos = np.arange(y0, ny, ystep)
  ylabels = ['%d' % (y*bin_size+min_size) for y in ylabel_pos]
  
  for label, anchor_map in anchor_maps.items():
    fig, ax = plt.subplots()
    fig.set_size_inches(8.0, 8.0)
    matshow_kw = {'norm':None, 'interpolation':'None', 'origin':'lower', 'vmin':0.0, 'vmax':anchor_map.max()}  
      
    anchor_map = anchor_map.astype(float)
    cax = ax.matshow(anchor_map, cmap=cmap, aspect='auto', **matshow_kw)
    
    ax.tick_params(which='both', direction='out', left=True, right=False, labelright=False, labelleft=True,
                   labeltop=False, labelbottom=True, top=False, bottom=True, pad=8)
                     
    ax.xaxis.set_ticks(xlabel_pos)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.yaxis.set_ticks(ylabel_pos)
    ax.set_yticklabels(ylabels, fontsize=9)

    ax.set_title('Anchor: ' + label)
    ax.set_xlabel('Separation from anchor (kb)')
    ax.set_ylabel('Fragment size (bp)')
    
    cbaxes = fig.add_axes([0.92, 0.25, 0.02, 0.5]) # LBWH
    cbar = plt.colorbar(cax, cax=cbaxes)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Read density/max', fontsize=8)
    
    save_path = '/data/dino_hi-c/MNase06_anchor_sep_size_' + label + '.pdf'
    
    plt.savefig(save_path, dpi=300)


if __name__ == '__main__':
   
  data_files = (#('DVNP', '/data/dino_hi-c/ChIP-seq_peaks/DVNP-HiC2_peaks.narrowPeak', 'BED', None, True),
                #('H2A', '/data/dino_hi-c/ChIP-seq_peaks/H2A-HiC2_peaks.narrowPeak', 'BED', None, True),
                ('Trinity_ssRNA_gene', '/data/dino_hi-c/hem_flye_4_ssRNA_Trinity.gff3', 'GFF', 'gene', False),
                #('Trinity_ssRNA_exon', '/data/dino_hi-c/hem_flye_4_ssRNA_Trinity.gff3', 'GFF', 'exon', False),
                )
  
  bam_path = '/data/dino_hi-c/SLX-17948_waller_hem_Mnase-06_sf.bam'
  #bam_path = 'SLX-17948_waller_hem_Mnase-06_sf.bam'
    
  anchor_dicts = _load_data_points(data_files)
  
  plot_bam_anchor_sep_size(bam_path, anchor_dicts, max_size=800, max_sep=5000)   
