import sys, os, time
import numpy as np
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict
from nuc_tools import util, io
from formats import bed, gff, ncc

def get_introns(gff_file):
  
  data_dict = gff.load_data_track(gff_file, ['exon','gene'])
  intron_dict = defaultdict(list)
  exon1_dict = defaultdict(list)
  exon2_dict = defaultdict(list)
  
  chromos = sorted(data_dict['gene'])
  
  for chromo in chromos:    
    genes = data_dict['gene'][chromo]
    exons = data_dict['exon'][chromo]
    
    pos_genes = genes[genes['strand'] == 1]
    pos_exons = exons[exons['strand'] == 1]
    pos_genes = pos_genes[pos_genes['pos2'].argsort()]
    pos_exons = pos_exons[pos_exons['pos2'].argsort()]
    
    neg_genes = genes[genes['strand'] == 0]
    neg_exons = exons[exons['strand'] == 0]
    neg_genes = neg_genes[neg_genes['pos2'].argsort()]
    neg_exons = neg_exons[neg_exons['pos2'].argsort()]
   
    epos2 =  pos_exons['pos2']
    epos1 =  pos_exons['pos1']
    
    eneg2 =  neg_exons['pos2']
    eneg1 =  neg_exons['pos1']

    gpos1 =  pos_genes['pos1']
    gneg1 =  neg_genes['pos1']
    
    pos_exon_gene_idx = np.searchsorted(pos_genes['pos2'], epos2)
    neg_exon_gene_idx = np.searchsorted(neg_genes['pos2'], eneg2)
    
    # + Strand
    
    
    pos2prev = -1
    j_prev = -1
    for i, pos2 in enumerate(epos2):
      pos1 = epos1[i]
      j = pos_exon_gene_idx[i]
      
      if j >= len(gpos1):
        break
        
      pos1gene = gpos1[j]
      
      if pos1 >= pos1gene:
        if j == j_prev:
          intron_dict[chromo].append((pos2prev, pos1))
          exon2_dict[chromo].append((pos1, pos2))
        else:
          exon1_dict[chromo].append((pos1, pos2))
        
      j_prev = j
      pos2prev = pos2
    
    
    # - strand
    
    pos2prev = -1
    j_prev = -1
    for i, pos2 in enumerate(eneg2):
      pos1 = eneg1[i]
      j = neg_exon_gene_idx[i]
      
      if j >= len(gneg1):
        break
        
      pos1gene = gneg1[j]
      
      if pos1 > pos1gene:
        if j == j_prev:
          intron_dict[chromo].append((pos2prev, pos1))
          exon2_dict[chromo].append((pos1, pos2))
        else:
          exon1_dict[chromo].append((pos1, pos2))
      
      j_prev = j
      pos2prev = pos2
            
  intron_point_dict = {}
  
  for chromo, data_list in intron_dict.items():
    pos = np.array(data_list, int)
    pos = pos.mean(axis=1).astype(int)
    intron_point_dict[chromo] = pos
  
  exon1_start_dict = {}
  exon1_end_dict = {}

  for chromo, data_list in exon1_dict.items():
    pos = np.array(data_list, int)
    exon1_start_dict[chromo] = pos[:,0]
    exon1_end_dict[chromo] = pos[:,1]
  
  exon2_start_dict = {}
  exon2_end_dict = {}

  for chromo, data_list in exon2_dict.items():
    pos = np.array(data_list, int)
    exon2_start_dict[chromo] = pos[:,0]
    exon2_end_dict[chromo] = pos[:,1]
  
  return intron_point_dict, exon1_start_dict, exon1_end_dict, exon2_start_dict, exon2_end_dict


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
  
@njit(int64[:,:](int64[:,:], int64[:,:], int64[:], int64, int64))
def _add_to_map2d(hic_pos, data_map, anchor_points, n_bins, bin_size):
  
  n_hic = len(hic_pos)
  n_points = len(anchor_points)
  lim = n_bins * bin_size
  j0 = 0
  
  for i in range(n_hic): # Ordered
    p1 = hic_pos[i,0]
    p2 = hic_pos[i,1]
    
    for j in range(j0, n_points): # Ordered
      a_pos = anchor_points[j]
      is_neg = a_pos < 0
      a_pos = abs(a_pos)
      a1 = a_pos-lim
      a2 = a_pos+lim
      if a2 < p1:
        j0 = j+1
      
      if  (a1 <= p1 < a2) and (a1 <= p2 < a2):
        delta_bins1 = (p1-a_pos) // bin_size # Negative if hi-c point is before anchor
        delta_bins2 = (p2-a_pos) // bin_size # Negative if hi-c point is before anchor
        
        if is_neg:
          k1 = n_bins - delta_bins1 - 1  # Zero/middle at n_bins
          k2 = n_bins - delta_bins2 - 1  
        else:
          k1 = n_bins + delta_bins1  # Zero/middle at n_bins
          k2 = n_bins + delta_bins2  
          
        data_map[k1,k2] += 1
        data_map[k2,k1] += 1
        
      elif a1 > p2:
        break
      
  return data_map
  
def plot_hic_anchor_density(ncc_path, anchor_dicts, anchor_styles, max_sep=5000, bin_size=20, close_thresh=400e3):
  
  contact_data1 = defaultdict(list)
  contact_data2 = defaultdict(list)
  
  with io.open_file(ncc_path) as file_obj:
   
    util.info('Reading {}'.format(ncc_path))
    n = 0
    
    for line in file_obj:
       chr_a, f_start_a, f_end_a, start_a, end_a, strand_a, \
       chr_b, f_start_b, f_end_b, start_b, end_b, strand_b, \
       ambig_code, pair_id, swap_pair = line.split()
       
       
       pos_a = int(f_end_a if strand_a == '+' else f_start_a)
       pos_b = int(f_end_b if strand_b == '+' else f_start_b)
       
       if (chr_a != chr_b) or (abs(pos_b-pos_a) > close_thresh):
         contact_data2[chr_b].append(pos_b)
         contact_data2[chr_a].append(pos_a)
      
       else:
         contact_data1[chr_b].append(pos_b)
         contact_data1[chr_a].append(pos_a)
       
       n += 1
       if n % 100000 == 0:
         util.info(' .. found {:,} contacts'.format(n), line_return =True)
         
         #if n > 1e7:
         #  break
   
  #util.info('Binning contacts')
  anchor_maps1 = {}
  anchor_maps2 = {}
  
  for label, data_dict in anchor_dicts.items():
    anchor_maps1[label] = np.zeros((max_sep // bin_size)*2, int)
    anchor_maps2[label] = np.zeros((max_sep // bin_size)*2, int)
    
  util.info('Aggregating maps')
  n_bins = max_sep // bin_size
  n_chromo = len(contact_data1)
  anchor_count = defaultdict(int)
  
  for c, chromo in enumerate(contact_data1):
    hic_pos1 = np.array(contact_data1[chromo])
    hic_pos1 = hic_pos1[hic_pos1.argsort()]
    hic_pos2 = np.array(contact_data2[chromo])
    hic_pos2 = hic_pos2[hic_pos2.argsort()]
    
    if c and c % 1000 == 0:
      util.info(f' .. {c:,} of {n_chromo:,}', line_return=True)
    
    for label, data_dict in anchor_dicts.items():
      if chromo not in data_dict:
        continue
      
      anchor_points = data_dict[chromo]
      anchor_points = anchor_points[np.abs(anchor_points).argsort()]
      anchor_count[label] += len(anchor_points)
      
      data_map = anchor_maps1[label] # Near
      _add_to_map(hic_pos1, data_map, anchor_points, n_bins, bin_size)

      data_map = anchor_maps2[label] # Far
      _add_to_map(hic_pos2, data_map, anchor_points, n_bins, bin_size)
  
  util.info(f' .. {c:,} of {n_chromo:,}', line_return=True)
  
  fig, (ax1, ax2) = plt.subplots(2, 1)
  fig.set_size_inches(8.0, 8.0)
       
  ax1.set_title(f'Anchor Hi-C distribution: near contacts ($< {close_thresh*1e-3:.1f}$ kb)')
  ax1.set_xlabel('Separation from anchor (bp)')
  ax1.set_ylabel(f'Mean Hi-C contact count ({bin_size} bp bins)')
  ax1.set_xlim((-max_sep, max_sep))

  ax2.set_title(f'Anchor Hi-C distribution: far contacts ($\u2265 {close_thresh*1e-3:.1f}$ kb, $trans$)')
  ax2.set_xlabel('Separation from anchor (bp)')
  ax2.set_ylabel(f'Mean Hi-C contact count ({bin_size} bp bins)')
  ax2.set_xlim((-max_sep, max_sep))
  #ax.set_ylim((0.8, 2.2))
  
  x_vals = np.linspace(-max_sep, max_sep, (2*max_sep)//bin_size)
  
  for label in anchor_maps1:
    color, linestyle = anchor_styles[label] 
    anchor_map = anchor_maps1[label]  .astype(float) # Near
    anchor_map /= anchor_count[label]    
    ax1.plot(x_vals, anchor_map, label=label, color=color, linestyle=linestyle, alpha=0.3)

    anchor_map = anchor_maps2[label] .astype(float) # Far
    anchor_map /= anchor_count[label]    
    ax2.plot(x_vals, anchor_map, label=label, color=color, linestyle=linestyle, alpha=0.3)
    
  #save_path = '/data/dino_hi-c/HiC_anchor_sep_distrib_' + label + '.pdf'
  #plt.savefig(save_path, dpi=300)
  
  ax1.axvline(0.0, linestyle='--', linewidth=0.5, alpha=0.5, color='#808080')
  ax2.axvline(0.0, linestyle='--', linewidth=0.5, alpha=0.5, color='#808080')
  ax1.legend(fontsize=8)
  ax2.legend(fontsize=8)
  
  plt.show()
  
  
def plot2D_hic_anchor_density(ncc_path, anchor_dicts, anchor_styles, max_sep=5000, bin_size=250):
  # Get local cis
  
  contact_data = defaultdict(list)
  t0 = time.time()
  
  with io.open_file(ncc_path) as file_obj:
   
    util.info('Reading {}'.format(ncc_path))
    n = 0
    
    for line in file_obj:
       chr_a, f_start_a, f_end_a, start_a, end_a, strand_a, \
       chr_b, f_start_b, f_end_b, start_b, end_b, strand_b, \
       ambig_code, pair_id, swap_pair = line.split()
       
       if chr_a != chr_b:
         continue
       
       pos_a = int(f_end_a if strand_a == '+' else f_start_a)
       pos_b = int(f_end_b if strand_b == '+' else f_start_b)
       
       if pos_a > pos_b:
         pos_a, pos_b = pos_b, pos_a
       
       contact_data[chr_a].append((pos_a, pos_b))
       
       n += 1
       if n % 100000 == 0:
         util.info(' .. found {:,} close contacts'.format(n), line_return =True)
         
         if n > 1e7:
           break
   
  n_bins =  max_sep // bin_size
  anchor_maps = {}
  
  for label, data_dict in anchor_dicts.items():
    anchor_maps[label] = np.zeros((n_bins*2, n_bins*2), int)
    
  util.info('Aggregating maps')
  n_bins = max_sep // bin_size
  n_chromo = len(contact_data)
  anchor_count = defaultdict(int)
  
  for c, chromo in enumerate(contact_data):
    hic_pos = np.array(contact_data[chromo])
    hic_pos = hic_pos[hic_pos[:,0].argsort()]
    
    if c and c % 1000 == 0:
      util.info(f' .. {c:,} of {n_chromo:,}', line_return=True)
    
    for label, data_dict in anchor_dicts.items():
      if chromo not in data_dict:
        continue
      
      anchor_points = data_dict[chromo]
      anchor_points = anchor_points[np.abs(anchor_points).argsort()]
      anchor_count[label] += len(anchor_points)
      data_map = anchor_maps[label]
      
      _add_to_map2d(hic_pos, data_map, anchor_points, n_bins, bin_size)
  
  util.info(f' .. {c:,} of {n_chromo:,}', line_return=True)

  util.info(' Time taken {:7.2f}s'.format(time.time()-t0))

  cmap = util.string_to_colormap('#FFFFFF,#80C0FF,#0080FF,#000000,#FF0000')
            
  xlabel_pos = ylabel_pos = np.arange(0, 2*n_bins+1, 20)
  xlabels = ylabels = ['%.1f' % ((x-n_bins)*bin_size*1e-3) for x in xlabel_pos]
    
  for label, anchor_map in anchor_maps.items():
    fig = plt.figure()
    fig.set_size_inches(8.0, 8.0)
    
    ax = fig.add_axes([0.09, 0.09, 0.8, 0.8]) # LBWH
    
    #anchor_map = np.log10(1.0 + anchor_map.astype(float))
    anchor_map = anchor_map.astype(float)
    matshow_kw = {'norm':None, 'interpolation':'None', 'origin':'lower', 'vmin':0.0, 'vmax':anchor_map.max()}  
    cax = ax.matshow(anchor_map, cmap=cmap, aspect='auto', **matshow_kw)
    
    ax.tick_params(which='both', direction='out', left=True, right=False, labelright=False, labelleft=True,
                   labeltop=False, labelbottom=True, top=False, bottom=True, pad=8)
                     
    ax.xaxis.set_ticks(xlabel_pos)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.yaxis.set_ticks(ylabel_pos)
    ax.set_yticklabels(ylabels, fontsize=9)

    ax.set_title(f'Aggregate Hi-C at anchor: {label}')
    ax.set_xlabel('Separation from anchor (kb)')
    ax.set_ylabel('Separation from anchor (kb)')
    ax.set_xlim((0, 2*n_bins))
    ax.set_ylim((0, 2*n_bins))
    
    ax.axvline(n_bins, linestyle='--', linewidth=0.5, alpha=0.75, color='#FF0000')
    ax.axhline(n_bins, linestyle='--', linewidth=0.5, alpha=0.75, color='#FF0000')

    cbaxes = fig.add_axes([0.90, 0.25, 0.02, 0.5]) # LBWH
    cbar = plt.colorbar(cax, cax=cbaxes)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f'Hi-C contacts [{bin_size} bp bins]', fontsize=8)
    
    save_path = f'/data/dino_hi-c/Dino_HiC_{label}_anchor_2Dmap_{n_bins}x{bin_size}bp.pdf'
    
    #plt.show()
    plt.savefig(save_path, dpi=300)


@njit(int64[:](int64[:], int64[:], int64[:], int64[:], int64, int64))
def _add_regions_to_map(starts, ends, data_map, anchor_points, n_bins, bin_size):
  
  n_regions = len(starts)
  n_points = len(anchor_points)
  lim = n_bins * bin_size
  j0 = 0
  
  for i in range(n_regions): # Ordered
    p1 = starts[i]
    p2 = ends[i]
    
    for j in range(j0, n_points): # Ordered
      a_pos = anchor_points[j]
      is_neg = a_pos < 0
      a_pos = abs(a_pos)
      a1 = a_pos-lim
      a2 = a_pos+lim
      if a2 < p1:
        j0 = j+1
      
      if  (a1 <= p1 < a2) or (a1 <= p2 < a2):
        if p1 < a1:
          p1 = a1
         
        if p2 >= a2:
          p2 = a2  
      
        delta_bins1 = (p1-a_pos) // bin_size # Negative if hi-c point is before anchor
        delta_bins2 = (p2-a_pos) // bin_size # Negative if hi-c point is before anchor
        
        if is_neg:
          k1 = n_bins - delta_bins1 - 1  # Zero/middle at n_bins
          k2 = n_bins - delta_bins2 - 1  
        else:
          k1 = n_bins + delta_bins1   # Zero/middle at n_bins
          k2 = n_bins + delta_bins2  
          
        data_map[k1:k2] += 1
        
      elif a1 > p2:
        break
      
  return data_map


def plot_bed_anchor_density(bed_paths, bed_labels, bed_colors, anchor_dicts, max_sep=4000, bin_size=50):
  
  data_maps = {}
  n_bins =  max_sep // bin_size
  anchor_keys = sorted(anchor_dicts)
  
  for key in anchor_keys:
    anchor_dict = anchor_dicts[key]
    data_maps[key] = {}
    
    for label, file_path in zip(bed_labels, bed_paths):
      data_dict = bed.load_data_track(file_path)
      chromos = sorted(data_dict)
      data_map = np.zeros(n_bins*2, int)
    
      for chromo in chromos:
        if chromo not in anchor_dict:
          continue
        
        anchor_points = anchor_dict[chromo]
        anchor_points = anchor_points[np.abs(anchor_points).argsort()]
        
        starts = data_dict[chromo]['pos1'].astype(int)
        ends = data_dict[chromo]['pos2'].astype(int)
        
        _add_regions_to_map(starts, ends, data_map, anchor_points, n_bins, bin_size)
        
      data_maps[key][label] = data_map
  
  n_rows = len(anchor_dicts)
  x_vals = np.linspace(-max_sep, max_sep, (2*max_sep)//bin_size)
  
  fig, axarr = plt.subplots(n_rows, 1, squeeze=False)
  
  for row, key in enumerate(anchor_keys):
     
    ax = axarr[row, 0]
    ax.set_title(f'Anchor: {key}')
    ax.set_xlim(-max_sep, max_sep-bin_size)
    ax.set_xlabel('Separation from anchor (bp)')
    ax.set_ylabel(f'Feature count [{bin_size} bp bins]')
    
    
    for label, color in zip(bed_labels, bed_colors):
      data_map = data_maps[key][label]
      
      ax.plot(x_vals, data_map, color=color, alpha=0.5, label=label)
    
    ax.legend(fontsize=8)  
  
  plt.show()

@njit(int64[:](int64[:], int64[:], int64[:], int64[:], int64[:], int64, int64))
def _add_points_to_prop_regions(starts, ends, strands, data_map, data_points, n_bins, overhang):
  
  n_regions = len(starts)
  n_points = len(data_points)
  j0 = 0
  
  for i in range(n_regions): # Ordered
    p1 = starts[i]
    p2 = ends[i]
    
    if (p2-p1) < n_bins:
      continue
    
    strand = strands[i]
    
    bin_size = float(p2-p1) / float(n_bins) 
    lim = int(overhang * bin_size)
    bin_size = int(bin_size)
    
    r1 = p1-lim
    r2 = p2+lim
    
    for j in range(j0, n_points): # Ordered
      pos = data_points[j]
      
      if pos < r1:
        j0 = j+1
      
      if r1 <= pos < r2:
        if strand: # + strand
          k = (pos-r1) // bin_size
        else:
          k = (r2-pos) // bin_size
        
        if k < n_bins + 2*overhang:
          data_map[k] += 1
      
      elif pos > r2: # Next points also out of region
        break
            
  return data_map


def plot_gff_regional_density(gff_path, ncc_path, bed_paths, bed_labels, bed_colors, 
                              gff_features=['gene','exon'], n_bins=25, overhang=10, close_thresh=400e3):
  
  contact_data1 = defaultdict(list)
  contact_data2 = defaultdict(list)
  
  with io.open_file(ncc_path) as file_obj:
   
    util.info('Reading {}'.format(ncc_path))
    n = 0
    
    for line in file_obj:
       chr_a, f_start_a, f_end_a, start_a, end_a, strand_a, \
       chr_b, f_start_b, f_end_b, start_b, end_b, strand_b, \
       ambig_code, pair_id, swap_pair = line.split()
       
       
       pos_a = int(f_end_a if strand_a == '+' else f_start_a)
       pos_b = int(f_end_b if strand_b == '+' else f_start_b)
       
       if (chr_a != chr_b) or (abs(pos_b-pos_a) > close_thresh):
         contact_data2[chr_b].append(pos_b)
         contact_data2[chr_a].append(pos_a)
      
       else:
         contact_data1[chr_b].append(pos_b)
         contact_data1[chr_a].append(pos_a)
       
       n += 1
       if n % 100000 == 0:
         util.info(' .. found {:,} contacts'.format(n), line_return =True)
  
  for chromo in contact_data1:
    contact_data1[chromo] = np.array(sorted(contact_data1[chromo]), int)
    
  for chromo in contact_data2:
    contact_data2[chromo] = np.array(sorted(contact_data2[chromo]), int)
         
  point_dicts  = {'Hi-C near':contact_data1, 'Hi-C far':contact_data2}
  point_colors = {'Hi-C near':'#FF2000',     'Hi-C far':'#0080FF'}
  
  for label, file_path, color in zip(bed_labels, bed_paths, bed_colors):
    util.info('Reading {}'.format(file_path))
    data_dict = bed.load_data_track(file_path)
    point_dict = {}
    
    for chromo in data_dict:
      data_array = data_dict[chromo]
      points = ((data_array['pos1'] + data_array['pos2']) // 2).astype(int)
      points = points[points.argsort()]
      
      point_dict[chromo] = points
    
    point_dicts[label] = point_dict
    point_colors[label] = color
    
  gff_dict = gff.load_data_track(gff_path, gff_features)
  data_maps = {}
   
  for feat in gff_dict:
    util.info(f'Mapping to {feat} features in {gff_path}')
    region_dicts = gff_dict[feat]
    data_maps[feat] = {}
    for label in point_dicts:
      data_maps[feat][label] = np.zeros(n_bins + overhang*2, int)
    
    for c, chromo in enumerate(region_dicts.keys()):
      if c and c % 100 == 0:
        util.info(f' .. {chromo}', line_return=True)
    
      data_array = region_dicts[chromo]
      starts = data_array['pos1'].astype(int)
      ends =   data_array['pos2'].astype(int)
      strands = data_array['strand'].astype(int)
      
      for label in point_dicts:
        point_dict = point_dicts[label]
        
        if chromo not in point_dict:
          continue
        
        points = point_dict[chromo] 
        _add_points_to_prop_regions(starts, ends, strands, data_maps[feat][label], points, n_bins, overhang)
  
  n_rows = len(gff_dict)
  
  pad = overhang/n_bins
  x_vals = np.linspace(-pad, 1.0+pad, n_bins+2*overhang)
  
  fig, axarr = plt.subplots(n_rows, 1, squeeze=False)
  
  for row, feat in enumerate(gff_dict):
     
    ax = axarr[row, 0]
    ax.set_title(f'Genome feature regions: {feat}')
    ax.set_xlim(-pad, 1.0+pad)
    
    ax.set_xlabel('Relative position in feature')
    ax.set_ylabel(f'Data fraction [{n_bins} regional bins 0-1]')
    
    
    for label in point_dicts:
      data_map = data_maps[feat][label].astype(float)
      data_map /= data_map.sum()
      
      ax.plot(x_vals, data_map, color=point_colors[label], alpha=0.5, label=label)
    
    ax.axvline(0.0, linestyle='--', linewidth=0.5, alpha=0.75, color='#808080')
    ax.axvline(1.0, linestyle='--', linewidth=0.5, alpha=0.75, color='#808080')
    ax.legend(fontsize=8)  
  
  plt.show()

    
  
    
if __name__ == '__main__':
  
  gff_path = '/data/dino_hi-c/hem_flye_4_ssRNA_Trinity.gff3'
  ncc_path = '/data/dino_hi-c/Hi-C_contact_data/SLX-17943_HEM_FLYE_4_Hi-C.ncc.gz'
   
  data_files = (#('DVNP',               ['#505050',], '/data/dino_hi-c/ChIP-seq_peaks/DVNP-HiC2_peaks.narrowPeak', 'BED', None),
                #('H2A',                ['#008000',], '/data/dino_hi-c/ChIP-seq_peaks/H2A-HiC2_peaks.narrowPeak', 'BED', None),
                ('Trin_gene', ['#FF8000','#B00000',], gff_path, 'GFF', 'gene'),
                ('Trin_exon', ['#0080FF','#000080'], gff_path, 'GFF', 'exon')
                )
 
  #anchor_dicts, anchor_styles = _load_data_points(data_files)
  
  #intron_point_dict, exon1_start_dict, exon1_end_dict, exon2_start_dict, exon2_end_dict = get_introns('/data/dino_hi-c/hem_flye_4_ssRNA_Trinity.gff3')
  
  #anchor_styles['first_exon_start'] = ('#4000FF', '-')
  #anchor_dicts['first_exon_start'] = exon1_start_dict
  #anchor_styles['other_exon_start'] = ('#FF8000', '-')
  #anchor_dicts['other_exon_start'] = exon2_start_dict
  
  #anchor_styles['intron_mid'] = ('#4000FF', '-')
  #anchor_dicts['intron_mid'] = intron_point_dict
   
 
  #plot_hic_anchor_density(ncc_path, anchor_dicts, anchor_styles, max_sep=4000, bin_size=50)
  #plot2D_hic_anchor_density(ncc_path, anchor_dicts, anchor_styles, max_sep=5000, bin_size=25)
  #plot2D_hic_anchor_density(ncc_path, anchor_dicts, anchor_styles, max_sep=1000, bin_size=10)
  
  #plot_bed_anchor_density(bed_paths, bed_labels, bed_colors, anchor_dicts, max_sep=12000, bin_size=25)

  bed_paths = ['/data/dino_hi-c/ChIP-seq_peaks/DVNP-HiC2_peaks.narrowPeak',
               '/data/dino_hi-c/ChIP-seq_peaks/H2A-HiC2_peaks.narrowPeak']
  bed_labels = ['DVNP','H2A']
  bed_colors = ['#505050','#008000']
  plot_gff_regional_density(gff_path, ncc_path, bed_paths, bed_labels, bed_colors)
 
