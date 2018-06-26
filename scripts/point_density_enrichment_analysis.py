from glob import glob
from math import sqrt
from numba import jit, int32, float64, int64
from random import randint
from scipy import stats
from collections import defaultdict

import math
import nuc_util as util
import numpy as np
import os

from matplotlib import pyplot as plt

@jit(int64[:](int64[:], int64[:,:], int64), cache=True)  
def points_region_interset(pos, regions, exclude=0):
  """
  Return an array of indices for points which either do (exclude=0) or do not
  (exclude=1) interest with ant array of regions 
  """
  
  sel_overlap = 1 - int(exclude)
  
  n = 0
  n_pos   = len(pos)
  n_reg   = len(regions)
  indices = np.empty(n_pos, int)
  order   = np.array(regions[:,0].argsort(), int)  
  
  for i in range(n_pos):
    
    if pos[i] < regions[order[0],0]:
      if not sel_overlap:
        indices[n] = i
        n += 1
      
      continue
      
    a = 0
    for k in range(n_reg):
      j = order[k]
      
      if (regions[j,0] <= pos[i]) and (pos[i] <= regions[j,1]):
        a = 1
        break
 
      if pos[i] < regions[j, 0]:
        break
        
    if sel_overlap == a:
      indices[n] = i
      n += 1
  
  return indices[:n]
 
  
@jit(float64[:](float64[:,:], float64[:,:], float64[:], int64[:], int64[:], int64[:], int64[:], int64, int64), cache=True)  
def get_point_density(coords_a, coords_b, values_b, seq_pos_a, seq_pos_b,
                      chrom_a, chrom_b, min_seq_sep, power):
  """
  Calculate 1/r^p spatial densties density for one set of coords based on another
  set of coords with asssociated signal values. Includes seq positions and
  chromosomes of coord points so sequentially close coordinates can be
  excluded.  
  """
  
  na, da = coords_a.shape
  nb, db = coords_b.shape
  pwr = -power/2.0
  
  densities = np.zeros(na, float)  
  
  for i in range(na):
    ix = coords_a[i, 0]
    iy = coords_a[i, 1]
    iz = coords_a[i, 2]
  
    for j in range(nb):
      if (chrom_a[i] != chrom_b[i]) or (abs(seq_pos_a[i] - seq_pos_b[j]) >= min_seq_sep):
        dx = ix - coords_b[j, 0]
        dy = iy - coords_b[j, 1]
        dz = iz - coords_b[j, 2]
 
        d = max(1.0, dx*dx + dy*dy + dz*dz) ** pwr

        densities[i] += values_b[j] * d

  return densities

  
@jit(float64[:](int64[:,:], float64[:], int64, int64, int64), cache=True)  
def bin_region_values(regions, values, bin_size=1000, start=0, end=-1):
  """
  Bin input regions and asscociated values into a histogram of new, regular
  regions. Accounts for partial overlap using proportinal allocation.
  """
                    
  n = values.shape[0]
  
  if len(regions) != n:
    data = (len(regions), n)
    msg = 'Number of regions (%d) does not match number of values (%d)'
    raise Exception(msg % data) 
  
  if end < 0:
    end = bin_size * np.int32(regions.max() / bin_size)
  
  s = start/bin_size
  e = end/bin_size
  n_bins = 1+e-s
  
  hist = np.zeros(n_bins, float)
  
  for i in range(n):
    v = values[i]
    
    if regions[i,0] > regions[i,1]:
      p1 = regions[i,1] 
      p2 = regions[i,0]
    
    else:
      p1 = regions[i,0]
      p2 = regions[i,1]
    
    if end < p1:
      continue
    
    if start > p2:
      continue  
    
    b1 = p1 / bin_size
    b2 = p2 / bin_size
    r = float(p2-p1)
    
    if b1 == b2: # All in one bin
      if b1 < s:
        continue
      
      if b1 > e:
        continue
        
      hist[b1-s] += v

    else:
      
      for b3 in range(b1, b2+1): # Region ovelaps bins
        if b3 < s:
          continue
        
        if b3 >= e:
          break  
        
        p3 = b3 * bin_size
        p4 = p3 + bin_size
        
        if (p1 >= p3) and (p1 < p4): # Start of region in bin 
          f = float(p4 - p1) / r 
        
        elif (p2 >= p3) and (p2 < p4): # End of region in bin
          f = float(p2 - p3) / r 
          
        elif (p1 < p3) and (p2 > p4): # Mid region in bin
          f = bin_size / r
        
        else:
          f = 0.0
        
        hist[b3-s] += v * f
  
  return hist
  

def get_point_density_enrichment(data_region_dict, data_value_dict, seq_pos_dict,
                                 coords_dict, a_region_dict, b_region_dict,
                                 min_seq_sep=int(350e3), power=3, n_perm=10, models=None):
  """
  Calculate spatial point density enrichment of a specified data track (regions
  and assoc. values) in a specified 3D coordinate set using a specified power
  law. Compares densities to  a null hypothesis with random circular permutation
  of chromosomal seq positions.
  """
  
  chromos = sorted(seq_pos_dict)
  chromo_idx = {chromo:i for i, chromo in enumerate(chromos)}
  
  a, b = seq_pos_dict[chromos[0]][:2]
  particle_size = b-a
  
  if not models:
    models = list(range(len(coords_dict[chromos[0]])))
  
  # Get flat arrays for all particles from separate chromosomes  
  coords = np.concatenate([coords_dict[c] for c in chromos], axis=1)
  seq_pos = np.concatenate([seq_pos_dict[c] for c in chromos], axis=0)
  int_chromos = np.concatenate([np.full(len(seq_pos_dict[c]), chromo_idx[c], int) for c in chromos], axis=0)
  n_coords = coords.shape[1]
  
  chromo_limits = {}
  for chromo in seq_pos_dict:
    pos = seq_pos_dict[chromo]
    start = particle_size * int(pos[0]/particle_size)
    end   = particle_size * int(math.ceil(pos[-1]/particle_size))
    chromo_limits[chromo] = (start, end)
    
  data_coords = []
  data_values = []
  data_seq_pos = []
  data_int_chromos = []
  data_idx = {}
  
  a_sizes = {}
  b_sizes = {}
  a_particles = {}
  b_particles = {}
  a_idx = {}
  b_idx = {}
  a_values = {}
  b_values = {}
  
  # Get flat arrays of track data, using same regions as particle arrays
  for chromo in chromos:
    start, end = chromo_limits[chromo]
    hist = bin_region_values(data_region_dict[chromo],
                             data_value_dict[chromo],
                             particle_size, start, end)
    idx  = hist.nonzero()[0]
    values = hist[idx]
    
    data_idx[chromo] = idx
    data_values.append( values )
    data_coords.append( coords_dict[chromo][:,idx] )
    data_seq_pos.append( seq_pos_dict[chromo][idx] )
    data_int_chromos.append( np.full(len(idx), chromo_idx[chromo], int) )
       
    centers = np.arange(start+particle_size/2, end+particle_size/2, particle_size)
    
    a_particles[chromo] = points_region_interset(centers, a_region_dict[chromo])
    b_particles[chromo] = points_region_interset(centers, b_region_dict[chromo])
    
    a_sizes[chromo] = len(a_particles[chromo])
    b_sizes[chromo] = len(b_particles[chromo])
    
    a_idx[chromo] = hist[a_particles[chromo]].nonzero()[0] # Where in the concatenated regions the particles lie
    b_idx[chromo] = hist[b_particles[chromo]].nonzero()[0]
    
  data_coords = np.concatenate(data_coords, axis=1)
  data_values = np.concatenate(data_values, axis=0)
  data_seq_pos = np.concatenate(data_seq_pos, axis=0)
  data_int_chromos = np.concatenate(data_int_chromos, axis=0)
  
  # Calc observed spatial density
  densities = np.zeros(n_coords)
  
  for model in models:
    densities += get_point_density(coords[model], data_coords[model], data_values, seq_pos,
                                   data_seq_pos, int_chromos, data_int_chromos,
                                   min_seq_sep, power)
  
  densities /= float(len(models))
  
  # Calc null, permutes spatial densities
  densities_null = np.zeros(n_coords)
  
  
  # Need to keep the proportion in A/B constant
  # - Permute around spliced, contigous regions
    
  for i in range(n_perm):
    util.info(' .. permutation %d' % (i+1))
    
    null_data_coords = []
    null_data_seq_pos = []
    
    for chromo in chromos:
      na = a_sizes[chromo]
      offset = randint(1, na-1)
      idx_a = (a_idx[chromo] + offset) % na
    
      nb = b_sizes[chromo]
      offset = randint(1, nb-1)
      idx_b = (b_idx[chromo] + offset) % nb

      # Convert indices in the concatenated A/B regions to indices in the whole chromo
      
      orig_a = a_particles[chromo][idx_a] # Select indices of A particles
      orig_b = b_particles[chromo][idx_b] # Select indices of B particles
      
      idx = np.sort(np.concatenate([orig_a, orig_b]))
      
      null_data_coords.append(coords_dict[chromo][:,idx])
      null_data_seq_pos.append(seq_pos_dict[chromo][idx])
      
    null_data_coords = np.concatenate(null_data_coords, axis=1)
    null_data_seq_pos = np.concatenate(null_data_seq_pos, axis=0)
    
    for model in models:
      densities_null += get_point_density(coords[model], null_data_coords[model], data_values,
                                          seq_pos, null_data_seq_pos, int_chromos,
                                          data_int_chromos, min_seq_sep, power)

  densities_null /= float(n_perm*len(models))
  nz = (densities * densities_null).nonzero()
  
  # Calc enrichment as log ratio
  enrichments = np.zeros(densities.shape, float)
  enrichments[nz] = np.log2(densities[nz]/densities_null[nz])
  
  # Unpack flat arrays into separate chromosomes
  chromo_enrichments = {}
  enrich_seq_pos = {}
  particle_regions = {}
  
  i = 0
  for chromo in chromos:
    pos = seq_pos_dict[chromo]
    n = len(pos)
    chromo_enrichments[chromo] = enrichments[i:i+n]
    particle_regions[chromo] = np.array([pos, pos+particle_size-1]).T
    i += n
  
  return particle_regions, chromo_enrichments


def calc_point_density_enrichment(input_paths, a_comp_bed_path, b_comp_bed_path, out_dir,
                                  min_seq_sep=350000, power=3, n_perm=10):
  """
  Calculate point spatial density enrichment for a set of structures.
  Results are save in a spcified directory
  """
  
  bed_data_path, n3d_coord_path = input_paths
  
  region_dict, value_dict, label_dict = util.load_bed_data_track(bed_data_path) 
  
  a_region_dict, a_value_dict, a_label_dict = util.load_bed_data_track(a_comp_bed_path)
  b_region_dict, b_value_dict, b_label_dict = util.load_bed_data_track(b_comp_bed_path)
  
  util.info('Working on structure %s' % n3d_coord_path)
  
  struc_file_root = os.path.splitext(os.path.basename(n3d_coord_path))[0]
  
  bed_file_root = os.path.splitext(os.path.basename(bed_data_path))[0]
  
  file_name = '%s_%s_sde.bed' % (bed_file_root, struc_file_root)
  out_bed_path = os.path.join(out_dir, file_name)
  
  if os.path.exists(out_bed_path):
    util.info('  .. found %s' % out_bed_path)
    return out_bed_path
  
  seq_pos_dict, coords_dict = util.load_n3d_coords(n3d_coord_path)
  
  particle_regions, chromo_enrich = get_point_density_enrichment(region_dict, value_dict, seq_pos_dict,
                                                                 coords_dict, a_region_dict, b_region_dict,
                                                                 min_seq_sep, power, n_perm)
  
  util.save_bed_data_track(out_bed_path, particle_regions, chromo_enrich)
  
  util.info('  .. saved %s' % out_bed_path)
  
  return out_bed_path


def get_structure_chromo_limits(n3d_coord_paths):
  """
  Get the sequence limits of particles for chromosomes in a structure
  """
  
  chromo_limits = {}
  bin_size = None
  
  for n3d_coord_path in n3d_coord_paths:
    seq_pos_dict, coords_dict = util.load_n3d_coords(n3d_coord_path)
  
    for chromo in seq_pos_dict:
      if not bin_size:
        a, b, = seq_pos_dict[chromo][:2]
        bin_size = b-a
 
      p1 = seq_pos_dict[chromo].min()
      p2 = seq_pos_dict[chromo].max()
      p1 = bin_size * int(p1/bin_size)
      p2 = bin_size * (1+int(math.ceil(p2/float(bin_size))))
 
      if chromo in chromo_limits:
        q1, q2 = chromo_limits[chromo]
        chromo_limits[chromo] = min(p1, q1), max(p2, q2)
      else:
        chromo_limits[chromo] = p1, p2
      
  return chromo_limits, bin_size
  

def split_bed_by_label(file_path, out_dir, file_tag='TF_BS'):
  """
  Split a BED file into separate files according to common row labels
  """
  
  out_files = {}

  with open(file_path) as file_obj:
    for line in file_obj:
      chr_a, pos_a, pos_b, label, val = line.split()
 
      if label in out_files:
        out_file_obj = out_files[label]
      else:
        out_file = 'split_%s_%s.bed' % (file_tag, label)
        out_path = os.path.join(out_dir, out_file)
        out_file_obj = open(out_path, 'w')
        out_files[label] = out_file_obj
 
      out_file_obj.write(line)

  for label in out_files:
    out_files[label].close()



def plot_data_track_correlations(paired_bed_files, ab_group_dict, bin_size=100000,
                                 quantplot=False, n_quant_bins=10):
  
  """
  Main graphing routine to display the correlations between the values in aligned
  (i.e. same binned region) pairs of named data tracks. Takes a group dict to
  analyse combined track pairs.
  """
  
  chromos = set()
  chromo_limits = {}
  
  group_names = sorted(paired_bed_files)
  
  # Load pairs of BED files into memory, get chromosome limits
  
  data_dict = {}
  tf_names = set()
  
  for group_name in paired_bed_files:
    for tf_name, bed_file_a, bed_file_b in paired_bed_files[group_name]:
      tf_names.add(tf_name)
 
      for bed_data_path in (bed_file_a, bed_file_b):
        if bed_data_path in data_dict:
          continue
 
        region_dict, value_dict, label_dict = util.load_bed_data_track(bed_data_path)
        chromos.update(region_dict.keys())
        data_dict[bed_data_path] = (region_dict, value_dict)
 
        for chromo in region_dict:
 
          a = region_dict[chromo].min()
          b = region_dict[chromo].max()
          a = bin_size * int(a/bin_size)
          b = bin_size * (1+int(math.ceil(b/float(bin_size))))
 
          if chromo in chromo_limits:
            c, d = chromo_limits[chromo]
            chromo_limits[chromo] = min(a, c), max(b, d)
          else:
            chromo_limits[chromo] = a, b
   
  # Group data into consistent bins spanning all chromsomes
  # - won't affect data that is aready binned at the same resolution
  # - but ensures all start and end positions are the same across all
  #   structures/datasets
  
  for bed_data_path in data_dict:
    region_dict, value_dict = data_dict[bed_data_path]
    chromo_hists = {}
    chromo_quants = {}
    
    hist_data = []
    hist_data2 = []
    offsets = []
    i = 0
    n = 0
    
    for chromo in region_dict:
      start, end = chromo_limits[chromo] 
      n += len(value_dict[chromo].nonzero()[0])
            
      hist = bin_region_values(region_dict[chromo], value_dict[chromo], bin_size, start, end)
      hist_data.append(hist)

      span = len(hist)
      offsets.append((chromo, i, span))
      i += span
    
    # Original, non-normalised binned data
    orig_data = np.concatenate(hist_data, axis=0)
   
    if "_sde" in bed_data_path: # A spatial density track
      hist_data = np.array(orig_data)
      
    else:
      # Simple rank normalisation for sequence density tracks
      hist_data = stats.rankdata(orig_data, method='dense').astype(float)
      hist_data /= hist_data.max()
          
    data_dict[bed_data_path] = hist_data, orig_data, n
  
  # Sub-function make box plots easier
  
  def ax_quantplot(ax, data, group=0, n_groups=1, text='', percentiles=(25.0, 75.0)):
    color = colors[group%len(colors)]
    
    x_vals = []
    y_vals = []
    y_errs = []
    y_lower = []
    y_upper = []
    
    for i, vals in enumerate(data):
      n = len(vals)
      
      if n < 5:
        continue
      
      m = np.mean(vals)
      
      vals = np.array(vals)
      
      #sem = 1.4826 * np.median(np.abs(vals-m))/(n**0.5)
      sem = np.std(vals, ddof=1)/(n**0.5)
      lower, upper = np.percentile(vals, percentiles)
      
      x_vals.append(i+0.5)
      y_vals.append(m)
      y_errs.append(sem)
      y_lower.append(lower)
      y_upper.append(upper)
          
    ax.plot(x_vals, y_vals, color=color, alpha=0.5, linewidth=2)
    ax.errorbar(x_vals, y_vals, y_errs, alpha=0.5, color=color)
    ax.scatter(x_vals, y_upper, color=color, alpha=0.4, s=10, marker='^')
    ax.scatter(x_vals, y_lower, color=color, alpha=0.4, s=10, marker='v')
    
    ax.text(0.1, 2.2-(0.3 * j), corr_text, color=color)
    
    """
    boxprops = {'linewidth':2, 'color':color}
    flierprops = {'marker':'.', 'color':'#808080', 'markersize':2, 'linestyle':'none'}
    meanprops = dict(marker='x', linewidth=2, markeredgecolor='black', markerfacecolor='black')
    whiskerprops = {'linestyle':'-', 'linewidth':2, 'color':color}
    medianprops = {'linewidth':1, 'color':'black'}
    capprops= {'linewidth':2, 'color':color}
    
    pos = range(group, len(data)*n_groups, n_groups)
    
    bp = ax.quantplot(data, positions=pos, whis=[10,90], widths=0.5, showfliers=False, 
                    showmeans=True, bootstrap=10, boxprops=boxprops,
                    flierprops=flierprops, medianprops=medianprops, meanprops=meanprops,
                    whiskerprops=whiskerprops, capprops=capprops)
    """
    
  
  # Constants
  ylabel = r' Normalised $1/{r^3}$ spatial density $log_{2}(\frac{Obs}{Exp})$'
  colors = ['#A0A000','#0080F0','#F04000','#00B000']
    
  tf_names = sorted(tf_names)
  n = len(tf_names)
  n_cols = min(n, 5)
  n_rows = int(math.ceil(n/float(n_cols)))
  
  # Plots for individual TFs/items
  
  fig, axarr = plt.subplots(n_rows, n_cols)
  
  row = 0
  col = 0
  n_groups = len(group_names)
  
  normed_gtf_data = {}
                
  for i, tf_name in enumerate(tf_names):
    
    if n_rows > 1:
      ax = axarr[row, col]
    elif n_cols > 1:
      ax = axarr[col]
    else:
      ax = axarr
        
    # Go through BED file pairs
    # - First should be binned sequence density
    # - Second should be spatiial density
    
    for j, group_name in enumerate(group_names):
      n_sites = 0
      x_data = []
      y_data = []
      x_data_orig = []
      y_data_orig = []
    
      for tf, bed_file_a, bed_file_b in paired_bed_files[group_name]:
        if tf == tf_name:
          # Sequence density
          ranked, orig, ca = data_dict[bed_file_a]
          x_data.append(ranked)
          x_data_orig.append(orig)
          n_sites += ca
 
          # Spatial density
          data, orig, cb = data_dict[bed_file_b]
          y_data.append(data)
          y_data_orig.append(orig)
 
      # Aggregate all pairs, over all cells/structures
      x_data = np.concatenate(x_data, axis=0)
      y_data = np.concatenate(y_data, axis=0)
      x_data_orig = np.concatenate(x_data_orig, axis=0)
      y_data_orig = np.concatenate(y_data_orig, axis=0)
 
      # Only consider non-zero sequence densities
      idx = x_data_orig.nonzero()
          
      x_data = x_data[idx]
      y_data = y_data[idx]
 
      order_x = x_data.argsort()
 
      box_data = y_data[order_x] # Arrange Y values in order of increasing X
      box_data = np.array_split(box_data, 4)
 
      # Z-normalise spatial density using values corresponding to the
      # first quartile of sequence density
      # - this reflects the nest random normal distrinb
 
      med = np.mean(box_data[0])
      std = np.std(box_data[0])
 
      y_data -= med
      y_data /= std
 
      # Store Z params for use later in combined plots
      normed_gtf_data[(group_name, tf_name)] = (x_data, y_data)

      x_data_orig = x_data_orig[idx]
      y_data_orig = y_data_orig[idx]
    
      r = np.corrcoef(x_data_orig, y_data_orig)[0,1]
      corr_text = '{} :R={:.2f}, N={:,}'.format(group_name[:4].strip(), r, n_sites)
 
      if quantplot:
        box_data = y_data[order_x] # Arrange Y values in order of increasing X
        box_data = np.array_split(box_data, n_quant_bins) # Separate data for each box 
        ax_quantplot(ax, box_data, j, n_groups, corr_text)
 
      else:
        ax.hexbin(x_data, y_data, cmap='Blues', bins='log', gridsize=50)
        ax.text(-9.0, 4.0, corr_text)
        
    ax.set_title(tf_name)
      
    if quantplot:
      ax.set_ylim([-1.1, 2.8])
      ax.set_xlim([-0.1, n_quant_bins + 0.1])
      ax.xaxis.set_ticks(np.arange(1.0, n_quant_bins + 0.5))
      if row == n_rows-1:
        ax.set_xlabel('Sequence density decile bin')
      
      p1 = 100/n_quant_bins
      
      if row == n_rows-1:
        ax.set_xticklabels(['%d' % x for x in range(p1,100,p1)], fontsize=12)
      else:
        ax.set_xticklabels([])
      
      if col != 0:
        ax.set_yticklabels([])
        
    else:
      ax.set_xlim([0.0, 1.0])
      ax.set_ylim([-3.1, 5.8])
      if row == n_rows-1:
        ax.set_xlabel('$log_2(Sequence density)$')
      
    if (col == 0) and (row == n_rows/2):
      ax.set_ylabel(ylabel)
    
    col += 1
    
    if col >= n_cols:
      col = 0
      row += 1          
  
  #if title:
  #  plt.suptitle(title)
  
  plt.show()
  
  # Combined plots 
  
  # Make a new "all" category
  
  tfs = []
  for tf_name in ab_group_dict:
    tfs += ab_group_dict[tf_name]
 
  #ab_group_dict = {}
  del ab_group_dict['Polycomb']
  ab_group_dict['D_All'] = tfs
  
  n_rows = 1
  n_cols = len(ab_group_dict)
  
  fig, axarr = plt.subplots(n_rows, n_cols)
  
  row = 0
  col = 0
  
  for i, ab_group_name in enumerate(sorted(ab_group_dict)):
 
    if n_rows > 1:
      ax = axarr[row, col]
    elif n_cols > 1:
      ax = axarr[col]
    else:
      ax = axarr
      
    if quantplot:
       ax.set_ylim([-1.0, 2.5])
       p1 = 100/n_quant_bins
       ax.set_xticklabels(['%d' % x for x in range(p1,100,p1)], fontsize=12)
       ax.xaxis.set_ticks(np.arange(1.0, n_quant_bins + 0.5))
 
       if row == n_rows-1:
         ax.set_xlabel('Sequence density decile bin')
 
    else:
      ax.set_xlim([0.0, 1.0])
      ax.set_ylim([-4.0, 5.0])
 
      if row == n_rows-1:
        ax.set_xlabel('Sequence density rank')

      
    for j, group_name in enumerate(group_names):
      n_sites = 0
      x_data = []
      y_data = []
      x_data_orig = []
      y_data_orig = []
      
      for tf_name, bed_file_a, bed_file_b in paired_bed_files[group_name]:
        if tf_name in ab_group_dict[ab_group_name]:
          xx, yy = normed_gtf_data[(group_name, tf_name)]         
          # Combine already normalised data for each TF

          null, orig, ca = data_dict[bed_file_a]
          
          idx = orig.nonzero()
          x_data.append(np.array(xx))
          x_data_orig.append(np.array(orig))
          n_sites += ca

          null, orig, cb = data_dict[bed_file_b]
          y_data.append(np.array(yy))
          y_data_orig.append(np.array(orig))

      
      x_data = np.concatenate(x_data, axis=0)
      y_data = np.concatenate(y_data, axis=0)
      x_data_orig = np.concatenate(x_data_orig, axis=0)
      y_data_orig = np.concatenate(y_data_orig, axis=0)

      r = np.corrcoef(x_data, y_data)[0,1]
      corr_text = '{}: R={:.2f}, N={:,}'.format(group_name[:4].strip(), r, n_sites)
 
      if quantplot:
        order_x = x_data.argsort()
        box_data = y_data[order_x] # Arrange Y values in order of increasing X
        box_data = np.array_split(box_data, n_quant_bins)
        ax_quantplot(ax, box_data, j, n_groups, corr_text)
 
      else:
        hb = ax.hexbin(x_data, y_data, cmap='Blues', bins='log', label=group_name, gridsize=50)
        ax.text(0.7, 4.0, corr_text)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('$log_{10}(count)$')
    
    ax.set_title(ab_group_name)
    
    if (col == 0) and (row == n_rows/2):
      ax.set_ylabel(ylabel)
    
    col += 1
    
    if col >= n_cols:
      col = 0
      row += 1          
  
  #if title:
  #  plt.suptitle(title)
  

  plt.show()


def bin_data_track(bed_data_path, chromo_limits, out_dir, bin_size, smooth=False,
                   out_bed_path=None, include_regions=None, exclude_regions=None, intersect_width=2000):
  """
  Create a binned data track, stored as a BED file, from an input BED file and specified chromsome
  limits. Saves data in a specified directory. Can include or exclude data points accoring to any
  overlap with lists of other regions.
  """
  
  if not out_bed_path:
    file_root = os.path.splitext(os.path.basename(bed_data_path))[0]
    
    out_bed_path = '%s_%dkb_bin.bed' % (file_root, int(bin_size/1000))
    out_bed_path = os.path.join(out_dir, out_bed_path)
  
  region_dict, value_dict, label_dict = util.load_bed_data_track(bed_data_path) 
  bin_region_dict = {}
  bin_value_dict = {}
  delta = bin_size-1
  hist_data = []
  half_bin = bin_size/2
  
  for chromo in region_dict:
    start, end = chromo_limits[chromo]
    
    regions = np.array(region_dict[chromo])
    values = value_dict[chromo]
    
    if include_regions:
      # Keep only data points which intersect these regions
    
      for intersect_regions in include_regions:
        filter_regions = intersect_regions[chromo] + np.array([-intersect_width, intersect_width])
        idx = points_region_interset(regions.mean(axis=1), filter_regions)
        regions = regions[idx]
        values = values[idx]
    
    if exclude_regions:
      # Remove datapoints which intersect these regions
      
      for intersect_regions in exclude_regions:
        filter_regions = intersect_regions[chromo] + np.array([-intersect_width, intersect_width])
        idx = points_region_interset(regions.mean(axis=1), filter_regions, exclude=1)
        regions = regions[idx]
        values = values[idx]
      
    if smooth:
      # Widen the data point regions by half a bin width for smoothing
      d = (regions[:,1]-regions[:,0])/2
      regions[:,0] -= half_bin - d
      regions[:,1] += half_bin - d
    
    hist = bin_region_values(regions, values, bin_size, start, end)
    pos = np.arange(start, end, bin_size)
     
    bin_region_dict[chromo] = np.array([pos, pos+delta]).T
    bin_value_dict[chromo] = hist
    hist_data.append(hist)
  
  hist_data = np.concatenate(hist_data, axis=0)
  
  util.save_bed_data_track(out_bed_path, bin_region_dict, bin_value_dict)
  util.info('  .. saved %s' % out_bed_path)
  
  return out_bed_path
    

if __name__ == '__main__':
  
   
  
  # # #  Some constants

  min_seq_sep = 350000 # Must be more than 3 beads
  power = 3            # Density is from sum of 1/r^power
  n_perm = 100         # Random sequence permutation for density null
  recalc_seqd = False  # Set to True to recalculate sequence densities
  recalc_spatd = False # Set to True to recalculate spatial densities (slow)
  n_cpu = 12           # Num parallel CPU cores to use for parallel spatial density calculations
  
  
  # # #  Get file paths
  
  # A/B compartment regions
  a_comp_bed_path = 'bed/Comp_A.bed'
  b_comp_bed_path = 'bed/Comp_B.bed'
  
  # N3d format genome structure coordinate files
  n3d_coord_paths = glob('n3d/Cell[123456]_100kb_x10.n3d')
  
  # Combined TF site BED files for grouping
  comb_bed_paths = glob('bed/TF_*.bed')
  
  # Separated TF site BED files (one for each TFs)
  bed_data_paths = glob('bed/tf_site_bed/split_TF_BS_*.bed')
  
  # Split combined TF BED files if separate TFs not available
  if not bed_data_paths:
    for file_path in glob('bed/TF_*.bed'):
      split_bed_by_label(file_path, 'bed/tf_site_bed/')
      
    bed_data_paths = glob('bed/tf_site_bed/split_TF_BS_*.bed')
    
  
  
  # # #  Get TF category groups from combined BED files
  
  ab_group_dict = {}
  for file_path in comb_bed_paths:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    group_name = file_name[8:]
    region_dict, value_dict, label_dict = util.load_bed_data_track(file_path) 
    
    names = set()
    for chromo in label_dict:
      names.update(set(label_dict[chromo]))
   
    ab_group_dict[group_name] = sorted(names)   


  # # # Load BED files used for overlapping with TF sites
  
  prom_regions, prom_values, null = util.load_bed_data_track('bed/Promoters.bed') 
  k4me3_regions, k4me3_values, null = util.load_bed_data_track('bed/H3K4me3_hap_EDL.bed') 
  k4me1_regions, k4me1_values, null = util.load_bed_data_track('bed/H3K4me1_GEO.bed') 
  
  
  
  # # #  Sequence density
  
  # Create binned sequence density data tracks
  # - with and without promoter or enhancer intersections
  # - Only needs to be done if something has changed
  
  if recalc_seqd:
    chromo_limits, bin_size = get_structure_chromo_limits(n3d_coord_paths)
    for bed_data_path in bed_data_paths:
       # All sites, smoothed
       bin_data_track(bed_data_path, chromo_limits, 'bed/tf_binned_all_bed', bin_size, smooth=True)
       
       # Promoter sites
       bin_data_track(bed_data_path, chromo_limits, 'bed/tf_binned_prom_bed', bin_size, smooth=True,
                      include_regions=[prom_regions, k4me3_regions], exclude_regions=[k4me1_regions])
                      
       # Enhancer assoc sites
       bin_data_track(bed_data_path, chromo_limits, 'bed/tf_binned_enha_bed', bin_size, smooth=True,
                      include_regions=[k4me1_regions], exclude_regions=[prom_regions, k4me3_regions])
  
  # The groups of sequence density data sets to use and thier respective titles 
  
  seq_bin_bed_paths = [#('Enhancers', glob('bed/tf_binned_enha_bed/*_100kb_bin.bed')),
                       #('Promoters', glob('bed/tf_binned_prom_bed/*_100kb_bin.bed')),
                       ('All sites', glob('bed/tf_binned_all_bed/*_100kb_bin.bed'))]

  
  # # # Spatial density data
  
  # Calculate spatial density enrichments (slow) if needed
  # - Only needed if min_seq_sep, power or n_perm is changed  
  
  if recalc_spatd:
    out_dir = 'bed/tf_sde_bed'
    common_args = (a_comp_bed_path, b_comp_bed_path, out_dir, min_seq_sep, power, n_perm)
 
    job_data = []
 
    for bed_data_path in bed_data_paths:
      for n3d_coord_path in n3d_coord_paths:
        job_data.append((bed_data_path, n3d_coord_path))
 
    bed_sde_paths = util.parallel_split_job(calc_point_density_enrichment, job_data, common_args, n_cpu)
  
  else:
    # Use pre-calculated spatial density enrichment bed files
    # - One for each TF-Structure combination
    bed_sde_paths = glob('bed/tf_sde_bed/*_sde.bed')
  
  
  # # #  Collate files for analysis
  
  # Pair the sequence density TF BED files with the spatial density BED files
  #  - Each SD will be for a different structure but the seq density partner will be the same
   
  tf_bin_files = {}
  tf_sde_files = {}
  
  # Get seq density files and extract TF name  
  for group_name, file_paths in seq_bin_bed_paths:
    tf_bin_files[group_name] = {}
    
    for file_path in file_paths:
      file_name = os.path.basename(file_path)
      tf_name = file_name.split('_100kb')[0][12:]
      tf_bin_files[group_name][tf_name] = file_path
  
  # Get spatial density files and extract TF name
  for file_path in bed_sde_paths:

    file_name = os.path.basename(file_path)
    tf_name = file_name.split('_Cell')[0][12:] 
    
    if tf_name in tf_sde_files:
      tf_sde_files[tf_name].append(file_path)
    else:
      tf_sde_files[tf_name] = [file_path]
  
  # Pair files by TF name, and separated according to enhancer/promoter group etc.
  paired_bed_files = {}
  
  for group_name in tf_bin_files:
    paired_bed_files[group_name] = []
  
    for tf_name in tf_bin_files[group_name]:
      if tf_name in ab_group_dict['Polycomb']:
        continue
      
      if tf_name == 'Pol2':
        continue
      
      bin_file = tf_bin_files[group_name][tf_name]

      print group_name, tf_name
 
      for sde_file in sorted(tf_sde_files[tf_name]):
        paired_bed_files[group_name].append((tf_name, bin_file, sde_file))

  
  # # #  Actually plot the data
  
  plot_data_track_correlations(paired_bed_files, ab_group_dict, quantplot=False)
    
    
  # # #  To do
  
  # Average SD over all structural models 
  # Increase num null permutations
  

