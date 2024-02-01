import sys, os, time
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nuc_tools import io, util
from formats import sam, bed, gff
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from numba import njit, int64, float64, int32, prange

def _load_data_points(data_files):

  track_data_dicts = {}
  use_middle = {}
  
  for label, file_path, feature, is_middle in data_files:
    util.info('Loading {}'.format(file_path))
    use_middle[label] = is_middle
    _, file_ext = os.path.splitext(file_path)
    file_ext = file_ext.lower()
    
    if file_ext in ('.bed',):
      track_data_dicts[label] = bed.load_data_track(file_path)
 
    elif file_ext in ('.gff','.gtf','.gff3'):
      track_data_dicts[label] = list(gff.load_data_track(file_path, [feature]).values())[0]
    
    else:
      print(f'File extension "{file_ext}" not known; use .bed or .gff')
      
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


@njit(int64[:,:](int64[:,:], int64[:,:], int64[:], int64, int64, int64, int64))
def _add_to_map(regions, data_map, data_points, mid_col, n, min_size, bin_size):
  lim = mid_col-1
  n_regions = len(regions)
  n_points = len(data_points)
  p_start = 0
  
  for r in range(n_regions): # E.g. MNase
    r1 = regions[r,0]
    r2 = regions[r,1]
    row = int((r2-r1-min_size)//bin_size) # Sequence separation
    
    for a in range(p_start, n_points): # Anchors
      a0 = data_points[a]
      a1 = abs(a0)-lim
      a2 = abs(a0)+lim
      if a1 < r1 < a2:
        j = r1-a1
 
        if r2 < a2:
          k = r2-a1
        else:
          k = n
 
      elif a1 < r2 < a2:
        j = 0
        k = r2-a1
 
      elif a1 > r2: # Anchor points are sorted, next will be forther away
        break
 
      else:
        continue
          
      if a2 < r1:
        p_start = a
      
      if a0 < 0: # Neg strand:
        for i in range(n-k, n-j):
          data_map[row,i] += 1
      
      else:
        for i in range(j,k):
          data_map[row,i] += 1 
      
  return data_map
  

def plot_bam_anchor_sep_size(save_file_root, bam_path, anchor_dicts, min_size=130, max_size=1000, max_sep=2000, bin_size=5):
 
  pdf_path = save_file_root + '_anchor_sep_vs_mol_size.pdf'
  
  pdf = PdfPages(pdf_path)
  
  nx = 2*max_sep + 1
  ny = int((max_size-min_size)//bin_size)+bin_size
  mid_col = max_sep + 1
  
  anchor_maps = {label:np.zeros((ny, nx), int) for label in anchor_dicts}
  
  t0 = time.time()
  #cmap = util.string_to_colormap('#FFFFFF,#0080FF,#000000,#FF0000')
  cmap = util.string_to_colormap('#FFFFFF,#004080,#FF0000') 
 
  chromo_regions = defaultdict(list)
    
  chromo_sizes = dict(sam.get_bam_chromo_sizes(bam_path))

  for i, (rname, sam_flag, chromo, pos, mapq, cigar, mate_chromo, mate_pos, t_len, seq, qual) in enumerate(sam.bam_iterator(bam_path)):
    
    if i % 100000 == 0:
      util.info('  .. {:,} {:7.2f}s'.format(i, time.time()-t0), line_return=True)
    
    #if i > 5e6:
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
  
  anchor_count_dict = defaultdict(int)
  
  for chromo in chromo_regions:
    regions = np.array(sorted(chromo_regions[chromo]))
     
    for label, data_dict in anchor_dicts.items():
      if chromo not in data_dict:
        continue
      
      data_points = data_dict[chromo]
      data_points = data_points[np.abs(data_points).argsort()]
      data_map = anchor_maps[label]
      anchor_count_dict[label] += len(data_points)
      _add_to_map(regions, data_map, data_points, mid_col, nx, min_size, bin_size)  
 
  util.info(' Time taken {:7.2f}s'.format(time.time()-t0))
           
  
  xlabel_pos = np.arange(0, 2*max_sep+1, 500)
  xlabels = ['%.1f' % ((x-max_sep)*1e-3) for x in xlabel_pos]
  
  ystep = 100//bin_size
  y0  = ystep-divmod(min_size//bin_size, ystep)[1] # Offset to the next 100  
  ylabel_pos = np.arange(y0, ny, ystep)
  ylabels = ['%d' % (y*bin_size+min_size) for y in ylabel_pos]
  
  # Normalise
  for label, anchor_map in anchor_maps.items():
    anchor_maps[label] = anchor_map.astype(float)/anchor_count_dict[label] 
  
  anchor_max = np.median([anchor_map.max() for anchor_map in anchor_maps.values()])
  
  for label, anchor_map in anchor_maps.items():
    
    fig = plt.figure() 
    fig.set_size_inches(8.0, 8.0)
    ax = fig.add_axes([0.1, 0.1, 0.78, 0.78]) # LBWH
    matshow_kw = {'norm':None, 'interpolation':'None', 'origin':'lower', 'vmin':0.0, 'vmax':anchor_max}  
      
    cax = ax.matshow(anchor_map, cmap=cmap, aspect='auto', **matshow_kw)
    
    ax.tick_params(which='both', direction='out', left=True, right=False, labelright=False, labelleft=True,
                   labeltop=False, labelbottom=True, top=False, bottom=True, pad=8)
                     
    ax.xaxis.set_ticks(xlabel_pos)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.yaxis.set_ticks(ylabel_pos)
    ax.set_yticklabels(ylabels, fontsize=9)

    ax.set_title(f'Anchor: {label} (n={anchor_count_dict[label]:,})')
    ax.set_xlabel('Separation from anchor (kb)')
    ax.set_ylabel('Fragment size (bp)')
    
    cbaxes = fig.add_axes([0.90, 0.25, 0.02, 0.5]) # LBWH
    cbar = plt.colorbar(cax, cax=cbaxes)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f'Read density {bin_size} bp bins (counts/anchor)', fontsize=8)
    
    pdf.savefig(dpi=300)
    plt.close(fig)
    
  print(f'Saved PDF {pdf_path}')
  pdf.close()
  plt.close()


if __name__ == '__main__':
  
  # Plot paramaters
  # - max mol size/sparation, max separation from anchor point, pixel bin size (bp)
  plotkw = {'max_size':800, 'max_sep':2000, 'bin_size':2}
  
  # The data_files defines regions on which to anchor molecule size analysis
  # - requires: text label, file path, file format (BED or GFF), GFF feature (if relevant), whether to use region middle as acnchor 
  
  # These lists are for anchor points, e.g. based on gene/intron/exon region classes
  bed_paths = ('/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2_intergenic.bed',
               '/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2_tiny_gene.bed',
               '/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2_short_gene.bed',
               '/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2_short_intron.bed',
               '/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2_short_first_exon.bed',
               '/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2_short_midd_exon.bed',
               '/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2_short_last_exon.bed',
               '/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2_intron.bed',
               '/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2_long_gene.bed',
               '/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2_first_exon.bed',
               '/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2_midd_exon.bed',
               '/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2_last_exon.bed',
               )
  
  # Label based on file name
  data_labels = [x.split('_v2_')[1][:-4] for x in bed_paths]
  
  nbed = len(bed_paths)
  gff_features = [None] * nbed
  
  # Use region middle only for 'intron' and 'intergenic'
  use_middle = [True if 'int' in x else False for x in data_labels]

  # data files to anchor dicts
  data_files = zip(data_labels, bed_paths, gff_features, use_middle)
  anchor_dicts = _load_data_points(data_files)
  
  # BAM files to analyse in terms of mol size and anchor separation
  bam_paths = ('/data/dino_hi-c/Data_tracks_read_align_BAM/ChIP-seq2/SLX-17946_waller_hem_H2A-HiC2_sf.bam',
               '/data/dino_hi-c/Data_tracks_read_align_BAM/ChIP-seq2/SLX-17946_waller_hem_DVNP-HiC2_sf.bam',
               '/data/dino_hi-c/SLX-17948_waller_hem_Mnase-15_sf.bam',
               '/data/dino_hi-c/SLX-17948_waller_hem_Mnase-06_sf.bam',
               '/data/dino_hi-c/SLX-17948_waller_hem_Mnase-02_sf.bam',
               )
  
  # Where to save plots, for each BAM  file
  out_plot_roots = ('/data/dino_hi-c/H2A',
                    '/data/dino_hi-c/DVNP',
                    '/data/dino_hi-c/MNase15',
                    '/data/dino_hi-c/MNase06',
                    '/data/dino_hi-c/MNase02')
  
  for plot_out, bam_path in zip(out_plot_roots, bam_paths):
    plot_bam_anchor_sep_size(plot_out, bam_path, anchor_dicts, **plotkw)   
  
  # This list of for anchor points from ChIP-seq
  data_files = (('DVNP', '/data/dino_hi-c/ChIP-seq_peaks/DVNP-HiC2_summits.bed', None, True),
                ('H2A',  '/data/dino_hi-c/ChIP-seq_peaks/H2A-HiC2_summits.bed',  None, True),
                )
  
  anchor_dicts = _load_data_points(data_files)
  
  # Same BAM paths to analyse
 
  out_plot_roots = ('/data/dino_hi-c/H2A_ChIP_peaks',
                    '/data/dino_hi-c/DVNP_ChIP_peaks',
                    '/data/dino_hi-c/MNase15_ChIP_peaks',
                    '/data/dino_hi-c/MNase06_ChIP_peaks',
                    '/data/dino_hi-c/MNase02_ChIP_peaks')
   
  for plot_out, bam_path in zip(out_plot_roots, bam_paths):
    plot_bam_anchor_sep_size(plot_out, bam_path, anchor_dicts, **plotkw)   
