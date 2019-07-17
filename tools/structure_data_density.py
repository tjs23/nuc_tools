import os, sys, math
import numpy as np
from time import time
from collections import defaultdict

from numba import jit, int32, float64, int64
from os.path import dirname
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Colormap
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial import distance

PROG_NAME = 'structure_data_density'
VERSION = '1.0.0'
DESCRIPTION = 'Measure the 3D/spatial density enrichment (eff. non-random clustering) of data ' \
              'tracks superimposed on 3D genome structures'
              
DEFAULT_MIN_PARTICLE_SEP = 3
DEFAULT_PDF_OUT = 'sdd_out_{}.pdf'
DEFAULT_MAX_RADIUS = 5.0
DEFAULT_POW = 3.0
COLORMAP_URL = 'https://matplotlib.org/tutorials/colors/colormaps.html'
MIN_FLOAT = sys.float_info.min
DENSITY_KEY = '_DENSITY_'
PDF_DPI = 200
A4_INCHES = (11.69, 8.27)
  
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
  
def get_point_density(d_mat, idx_a, idx_b, values):

  if idx_a is not None:
    d_mat = d_mat[idx_a]

  if idx_b is not None:
    d_mat = d_mat[:,idx_b]
  
  na, nb = d_mat.shape

  w_mat = np.ones(na)[:,None] * values
  
  densities = (d_mat * w_mat).sum(axis=1)
  
  return densities


def get_density_matrix(n3d_path, radius, min_seq_sep, power):
  
  from formats import n3d
  from scipy.spatial import cKDTree, distance
  from nuc_tools import util
    
  seq_pos_dict, coords_dict = n3d.load_n3d_coords(n3d_path)
  
  chromo_limits = {}
  
  for chromo in seq_pos_dict:
    pos = seq_pos_dict[chromo]
    particle_size = pos[1] - pos[0]
    start = particle_size * int(pos[0]/particle_size)
    end   = particle_size * int(math.ceil(pos[-1]/particle_size))
    chromo_limits[chromo] = (start, end)

  chromos = sorted(chromo_limits)
  chr_idx = {chromo:i for i, chromo in enumerate(chromos)}
  
  coords = np.concatenate([coords_dict[c] for c in chromos], axis=1)
  seq_pos = np.concatenate([seq_pos_dict[c] for c in chromos], axis=0)
  chr_idx = np.concatenate([np.full(len(seq_pos_dict[c]), chr_idx[c], int) for c in chromos], axis=0)
  n_models, n_coords, n_dim = coords.shape
  d_mat = np.zeros((n_coords, n_coords), 'float32')
  
  # Get points where mean coord is within bounds  
  kdtree = cKDTree(coords.mean(axis=0))
  idx_pairs = kdtree.query_pairs(radius, output_type='ndarray')
  idx_a, idx_b = idx_pairs.T
  
  # Keep if different chromos or large enough seq sep
  valid = (chr_idx[idx_a] != chr_idx[idx_b]) | (np.abs(seq_pos[idx_a]-seq_pos[idx_b]) >= min_seq_sep)
  idx_a = idx_a[valid]
  idx_b = idx_b[valid]
  
  # Average of densities over all models
  for i in range(n_models):    
    util.info('  .. distance matrix for model %d' % i, line_return=True)     
    p1 = coords[i, idx_a]
    p2 = coords[i, idx_b]
    
    d = p1-p2
    d = (d*d).sum(axis=1)
    d[d < 1.0] = 1.0
    d **= (-power/2.0)
    d_mat[idx_a, idx_b] += d/float(n_models)
  
  return d_mat, chromo_limits


def get_pde(dens_mat, chromo_limits, anchor_bed_path, density_bed_path, bin_size):
  """
  """
  from formats import n3d, bed
  from matplotlib import pyplot as plt
  
  # # # particle_size must be checked to be same in all structures
  # # # null hypothesis could include a max seq sep 
  
  anch_regions, anch_values = bed.load_bed_data_track(anchor_bed_path)[:2]
  
  if anchor_bed_path == density_bed_path:
    data_regions, data_values = anch_regions, anch_values
  else:
    data_regions, data_values = bed.load_bed_data_track(density_bed_path)[:2]
  
  chromos = sorted(chromo_limits)
      
  # Get binned data track values over universal/max extent chromosome
  
  data_hists = {}
  anch_hists = {}
  
  # Structures can have a different regions
  
  for chromo in chromo_limits:
    start, end = chromo_limits[chromo]
    data_hists[chromo] = bin_region_values(data_regions[chromo], data_values[chromo],
                                           bin_size, start, end)
    
    hist = data_values[chromo]
    anch_hists[chromo] = bin_region_values(anch_regions[chromo], anch_values[chromo],
                                           bin_size, start, end)
                         
  # Get flat arrays for all particles from separate chromosomes

  anch_idx = []
  data_idx = []
  data_values = []
  
  # Get flat arrays of track data, using same regions as particle arrays
  a = 0
  b = 0
  for chromo in chromos:    
    hist = data_hists[chromo]
    idx  = hist.nonzero()[0]
    data_idx.append(a + idx)
    a += len(hist)
    
    hist2 = anch_hists[chromo]
    idx2 = hist2.nonzero()[0]
    anch_idx.append(b +  idx2)
    b += len(hist2)
    
    data_values.append( hist[idx] )
  
  data_values = np.concatenate(data_values, axis=0)
  anch_idx = np.concatenate(anch_idx, axis=0)
  data_idx = np.concatenate(data_idx, axis=0)
  
  # Calc observed spatial density for this structure's particles
  dens_obs = get_point_density(dens_mat, anch_idx, data_idx, data_values)
  dens_exp = get_point_density(dens_mat, None, data_idx, data_values)

  return dens_obs, dens_exp
    
    
def correlation_plot(n_tracks, dens_exp, data_labels, pdf=None, cmap='Blues', split_idx=None, is_primary=True, max_dens=4.5, hist_bins2d=50):
  
  hist_range =  (0.0, max_dens)
  fig, axarr = plt.subplots(n_tracks, n_tracks, sharex=True, sharey=True)
  fig.set_size_inches(*A4_INCHES)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.05)
  
  corr_mat = np.zeros((n_tracks, n_tracks))
  
  if split_idx is None:
    plt.suptitle('Density correlations')
    
  elif split_idx and is_primary:
    plt.suptitle('Density correlations : primary structures')
    
  else:
    plt.suptitle('Density correlations : secondary structures')

  for row, col in dens_exp:

    if n_tracks > 1:
      ax = axarr[row, col]
    else:
      ax = axarr
 
    if split_idx is None:
      ref = np.concatenate(dens_exp[(row, row)]) # All particle
      exp = np.concatenate(dens_exp[(row, col)]) # All particle
 
    elif is_primary:
      ref = np.concatenate(dens_exp[(row, row)][:split_idx]) # All particle
      exp = np.concatenate(dens_exp[(row, col)][:split_idx]) # All particle
    
    else:
      ref = np.concatenate(dens_exp[(row, row)][split_idx:]) # All particle
      exp = np.concatenate(dens_exp[(row, col)][split_idx:]) # All particle
 
    nz = (ref > 0) & (exp > 0)

    x_vals = np.log10(ref[nz])
    y_vals = np.log10(exp[nz])

    r, p = stats.pearsonr(x_vals, y_vals)
    corr_mat[row, col] = r

    ax.hist2d(x_vals, y_vals,
              bins=hist_bins2d, range=(hist_range, hist_range),
              cmap=cmap)

    ax.plot([0.0, max_dens], [0.0, max_dens], color='#808080', alpha=0.5, linestyle='--', linewidth=0.5)

    ax.text(0.05, 0.95, '$\\rho$={:.3f}\n$n$={:,}'.format(r,len(x_vals)), transform=ax.transAxes,
            color='#404040', verticalalignment='top', alpha=0.5, fontsize=8)

            
    if row == 0:
      axr = ax.twiny()
      axr.set_xticks([])
      axr.set_xlabel(data_labels[col], fontsize=11)

    if col == n_tracks-1:
      axr = ax.twinx()
      axr.set_yticks([])
      axr.set_ylabel(data_labels[row], fontsize=11)

    if (row == n_tracks-1) and (col == int(n_tracks/2)):
      ax.set_xlabel('Density percentile', fontsize=11)

    if (col == 0) and (row == int(n_tracks/2)):
      ax.set_ylabel('Density percentile', fontsize=11)
  
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show()
    
  plt.close()
  
  return corr_mat

def density_plot(title, mat, val_label, data_labels, cmap, pdf, vmin=None, vmax=None):  
  
  mh = 0.5
  bw = 0.1
  dh = 0.2
  
  n_tracks = len(mat)
  
  if vmax is None:
    vmax = mat.max()
  
  if vmin is None:
    if mat.min() < 0:
      vmin = min(mat.min(), -vmax)
    else:
      vmin = 0.0
      
  fig = plt.figure()    
  fig.set_size_inches(8, 8)
  
  plt.suptitle(title)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.04, hspace=0.04)
  
  ax1 = fig.add_axes([bw,bw+dh,mh,mh])
  
  dist_mat = distance.pdist(mat)
  linkage = hierarchy.linkage(dist_mat, method='ward', optimal_ordering=True)
  order = hierarchy.leaves_list(linkage)
  
  xylabels = [data_labels[i] for i in order]
  xylabel_pos = np.linspace(0.0, n_tracks-1, n_tracks)
  
  cax1 = ax1.matshow(mat[order][:,order], cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
  ax1.set_xticklabels(xylabels, fontsize=8, rotation=90.0)
  ax1.set_yticklabels(xylabels, fontsize=8)

  ax1.xaxis.set_ticks(xylabel_pos)
  ax1.xaxis.set_tick_params(which='both', direction='out')
 
  ax1.yaxis.set_ticks(xylabel_pos)
  ax1.yaxis.set_tick_params(which='both', direction='out')

  dgax1 = fig.add_axes([bw, bw, mh, dh]) # left, bottom, w, h  
  ddict = hierarchy.dendrogram(linkage, orientation='bottom', labels=data_labels,
                               above_threshold_color='#000000', no_labels=True,
                               link_color_func=lambda k: '#000000', ax=dgax1)
  dgax1.set_xticklabels([])
  dgax1.set_xticks([])
  dgax1.set_axis_off()               
  
  cbax1 = fig.add_axes([bw+mh+0.05, bw+dh, 0.02, mh]) # left, bottom, w, h
  
  cbar = plt.colorbar(cax1, cax=cbax1, orientation='vertical')
  cbar.ax.tick_params(labelsize=8)
  cbar.set_label(val_label, fontsize=9)
  
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show() 
  
  plt.close()
  

def structure_data_density(struc_paths1, struc_paths2, data_tracks, data_labels=None,
                           out_path=None, screen_gfx=None, radius=DEFAULT_MAX_RADIUS, 
                           min_sep=DEFAULT_MIN_PARTICLE_SEP, dist_pow=DEFAULT_POW,
                           cmap=plt.get_cmap('Blues'), cache_dir=None):
  
  from nuc_tools import io, util
  from formats import n3d
  
  if out_path:
    out_path = io.check_file_ext(out_path, '.pdf')
  
  else:
    file_name = DEFAULT_PDF_OUT.format('m50Qk') # util.get_rand_string(5))
    dir_path = dirname(struc_paths1[0])
    out_path = os.path.join(dir_path, file_name)
  
  if struc_paths2:
    msg = 'Analysing {} data tracks comparing two groups with structures {} and {} respectively.'
    util.info(msg.format(len(data_tracks), len(struc_paths1), len(struc_paths2)))
  
  else:
    msg = 'Analysing {} data tracks with {} structures, using a random background.'
    util.info(msg.format(len(data_tracks), len(struc_paths1)))

  if data_labels:
    for i, label in enumerate(data_labels):
      data_labels[i] = label.replace('_',' ')
      
    while len(data_labels) < len(data_tracks):
      i = len(data_labels)
      data_labels.append(io.get_file_root(data_tracks[i]))
      
  else:
    data_labels = [io.get_file_root(x) for x in data_tracks]
  
  if cache_dir and not os.path.exists(cache_dir):
    io.makedirs(cache_dir, exist_ok=True)

  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(out_path)  
    
  # Get universal chromosome regions
  
  util.info('Getting chromosome limits')  
  chromo_limits, bin_size = n3d.get_chromo_limits(struc_paths1+struc_paths2)

  min_seq_sep = min_sep * bin_size
  n_tracks = len(data_tracks)

  dens_obs = defaultdict(list)
  dens_exp = defaultdict(list)
  
  for n3d_path in struc_paths1 + struc_paths2:
    util.info('Analysing structure {}'.format(os.path.basename(n3d_path)))  
    n3d_root = io.get_file_root(n3d_path)

    if cache_dir:
      cache_file = os.path.join(cache_dir, n3d_root + '_sd.npz')
      
      if os.path.exists(cache_file):
        file_dict = np.load(cache_file)
        struc_chromo_lims = {}
        
        for key in file_dict:
          if key == DENSITY_KEY:
            dens_mat = file_dict[key]
          else:
            struc_chromo_lims[key] = file_dict[key]
        
        util.info('  .. read {}'.format(cache_file))
       
      else:
        dens_mat, struc_chromo_lims = get_density_matrix(n3d_path, radius, min_seq_sep, dist_pow)
        ddict = struc_chromo_lims.copy()
        ddict[DENSITY_KEY] = dens_mat
        np.savez_compressed(cache_file, **ddict) 
        util.info('  .. cached {}'.format(cache_file))
        
    else:
      dens_mat, struc_chromo_lims = get_density_matrix(n3d_path, radius, min_seq_sep, dist_pow)
    
    for row, anchor_bed_path in enumerate(data_tracks): # Anchor
      util.info('  Considering {}'.format(os.path.basename(anchor_bed_path)))
      bed_root1 = io.get_file_root(anchor_bed_path)
      
      for col, density_bed_path in enumerate(data_tracks): # Data
        key = row, col
        util.info('  .. compared to {}'.format(os.path.basename(density_bed_path)))
        bed_root2 = io.get_file_root(density_bed_path)
        
        if cache_dir:
          cache_file =  os.path.join(cache_dir, '{}_{}_{}_sdd.npz'.format(n3d_root, bed_root1, bed_root2))

          if os.path.exists(cache_file):
            file_dict = np.load(cache_file)
            obs = file_dict['obs']
            exp = file_dict['exp']
            util.info('  .. read {}'.format(cache_file))
            
          else:
            obs, exp = get_pde(dens_mat, struc_chromo_lims,
                               anchor_bed_path, density_bed_path, bin_size)
            
            ddict = {'obs':obs, 'exp':exp}
            np.savez_compressed(cache_file, **ddict) 
            util.info('  .. cached {}'.format(cache_file))
          
        else:    
          obs, exp = get_pde(dens_mat, struc_chromo_lims,
                             anchor_bed_path, density_bed_path, bin_size)
 
        dens_obs[key].append(obs)
        dens_exp[key].append(exp)
        
  # Correlations
  
  hist_bins = 25
  hist_bins2d = 50
 
  if struc_paths2:
    split_idx = len(struc_paths1)
    corr_mat1 = correlation_plot(n_tracks, dens_exp, data_labels, pdf, cmap, split_idx, True)
    corr_mat2 = correlation_plot(n_tracks, dens_exp, data_labels, pdf, cmap, split_idx, False)
  else:
    corr_mat1 = correlation_plot(n_tracks, dens_exp, data_labels, pdf, cmap)
    
  # Enrichment distribs
  
  fig, axarr = plt.subplots(n_tracks, n_tracks, sharey=True)    
  fig.set_size_inches(*A4_INCHES)
  
  plt.suptitle('Density enrichments')
  plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.9, wspace=0.05, hspace=0.05)
  
  js_mat = np.zeros((n_tracks, n_tracks))
  pv_mat = np.zeros((n_tracks, n_tracks)) + MIN_FLOAT
  hist_range = (0.0, 100.0)
  exp_val = 1.0/hist_bins
  
  for key in dens_exp:
    row, col = key
    
    if n_tracks > 1:
      ax = axarr[row, col]
    else:
      ax = axarr
    
    if struc_paths2: # Compare two structure groups
      a = len(struc_paths1)
      exp1 = np.sort(np.concatenate(dens_exp[key][:a]))
      obs1 = np.concatenate(dens_obs[key][:a])
      obs1 = np.searchsorted(exp1, obs1).astype(float)
      n = float(len(exp1))      
      obs1 *= 100.0/(n+1.0)
      
      exp2 = np.sort(np.concatenate(dens_exp[key][a:]))
      obs2 = np.concatenate(dens_obs[key][a:])
      obs2 = np.searchsorted(exp2, obs2).astype(float)
      n = float(len(exp2))      
      obs2 *= 100.0/(n+1.0)
           
      hist_obs, edges = np.histogram(obs1, bins=hist_bins, range=hist_range)
      hist_exp, edges = np.histogram(obs2, bins=hist_bins, range=hist_range)
      
      label1 = 'Group1'
      label2 = 'Group2'
      
    else: # Compare vs random
      obs = np.concatenate(dens_obs[key])
      exp = np.sort(np.concatenate(dens_exp[key]))
      n = float(len(exp)) 
      
      obs = np.searchsorted(exp, obs).astype(float)
      obs *= 100.0/(n+1.0)
      
      exp = np.linspace(0.0, 100.0, hist_bins) 
      
      label1 = 'Obs'
      
      # Log
      #obs = np.log10(obs[obs > 0])
      #exp = np.log10(exp[exp > 0])
 
      hist_obs, edges = np.histogram(obs, bins=hist_bins, range=hist_range)
      hist_exp, edges = np.histogram(exp, bins=hist_bins, range=hist_range)
    
    n_obs = hist_obs.sum()
    x_vals = np.linspace(0.0, 100.0, hist_bins) 
    
    nz = (hist_obs > 0) & (hist_exp > 0)

    hist_obs = hist_obs[nz].astype(float)
    hist_exp = hist_exp[nz].astype(float)
    
    hist_exp *= n_obs/hist_exp.sum()
    
    z, pv = stats.power_divergence(hist_obs, hist_exp, lambda_=0)
    pv = max(pv, MIN_FLOAT)
    
    x_vals = x_vals[nz]
    
    hist_obs /= float(hist_obs.sum())
    hist_exp /= float(hist_exp.sum())
       
    ax.plot(x_vals, hist_obs, color='#0080FF', label=label1)
    
    if struc_paths2:
      ax.plot(x_vals, hist_exp, color='#808080', label=label2)
    
    ax.plot((0.0, 100.0), (exp_val, exp_val), color='#808080', ls='--', alpha=0.5, label='Exp')
    
    js = 0.5 * (hist_obs * np.log(hist_obs/hist_exp)).sum()
    #js += 0.5 * (hist_exp * np.log(hist_exp/hist_obs)).sum()
    
    pv_mat[row, col] = pv
    js_mat[row, col] = js
    
    y_max = 1.4 * exp_val

    ax.text(0.05, 0.95, 'JS={:.3f}\np={:.3e}'.format(js,pv),
            color='#404040', verticalalignment='top',
            alpha=0.5, fontsize=8, transform=ax.transAxes)
    
    if row == 0:
      axr = ax.twiny()
      axr.set_xticks([])
      axr.set_xlabel(data_labels[col], fontsize=8)      
    
    if row < (n_tracks-1):
      ax.set_xticks([])

    if (row == n_tracks-1) and (col == int(n_tracks/2)):
      ax.set_xlabel('Density percentile', fontsize=11)
    
    if col == n_tracks-1:
      axr = ax.twinx()
      axr.set_yticks([])
      axr.set_ylabel('{}\nn={:,}'.format(data_labels[row], n_obs), fontsize=8)
     
    if (col == 0) and (row == int(n_tracks/2)):
      ax.set_ylabel('Fraction total', fontsize=11)
  
  ax = fig.add_axes([0.1, 0.9, 0.1, 0.1])
  ax.plot([], [], color='#0080FF', label=label1)
    
  if struc_paths2:
    ax.plot([], [], color='#808080', label=label2)
    
  ax.plot([], [], color='#808080', ls='--', alpha=0.5, label='Exp')
  ax.axis('off')
  ax.legend(fontsize=11, frameon=False)
    
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show() 
  
  plt.close()

  cmap2 = LinearSegmentedColormap.from_list(name='pcm', colors=['#0060E0','#FFFFFF','#E02000'], N=255)    
  
  js_mat += js_mat.T
  
  density_plot('Density enrichment - all stuctures', js_mat, 'Jensen-Shannon divergence', data_labels, cmap2, pdf, 0.0, 0.1)

  density_plot('Density correlation', corr_mat1, 'Pearson correlation coefficient', data_labels, cmap2, pdf, 0.0, 1.0)

  if pdf:
    pdf.close()
    util.info('Written {}'.format(out_path))
  else:
    util.info('Done')
  
                             
def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(metavar='N3D_FILES', nargs='+', dest='i',
                         help='One or more genome structure files in N3D format. Accepts wildcards.')

  arg_parse.add_argument('-s', '--structs2', metavar='N3D_FILES', nargs='+', dest='s',
                         help='One or more secondary genome structure files in N3D format. Accepts wildcards. ' \
                              'If not specified comparison will be made to a random null hypothesis, rather ' \
                              'than between structure groups.')

  arg_parse.add_argument('-d', '--data-tracks', metavar='BED_FILES', nargs='+', dest='d',
                         help='One or more data track files in BED format. Accepts wildcards.')

  arg_parse.add_argument('-l', '--data-labels', metavar='DATA_NAMES', nargs='+', dest='l',
                         help='Optional textual labels/names for the input data tracks.')

  arg_parse.add_argument('-g', '--screen-gfx', default=False, action='store_true', dest='g',
                         help='Display graphics on-screen using matplotlib, where possible and ' \
                              'do not automatically save graphical output to file.')

  arg_parse.add_argument('-m', '--min_particle-sep', default=DEFAULT_MIN_PARTICLE_SEP,
                         metavar='MIN_PARTICLE_SEP', type=int, dest="m",
                         help='The minimum separation of  structure particles for analysis. ' \
                              'Avoids linear sequence correlation effects and distortions due to the ' \
                              'courseness of the 3D model. Default: %d particles' % DEFAULT_MIN_PARTICLE_SEP)

  arg_parse.add_argument('-r', '--max_radius', default=DEFAULT_MAX_RADIUS,
                         metavar='MAX_SERACH_RADIUS', type=float, dest="r",
                         help='The maximum search distance within which data densties are calculated ' \
                              'for a genome structure point. Default: %.2f' % DEFAULT_MAX_RADIUS)

  arg_parse.add_argument('-o', '--out_pdf', metavar='OUT_PDF_FILE', default=None, dest='o',
                         help='Optional output PDF file name. If not specified, a default will be used.')

  arg_parse.add_argument('-p', '--power_law', default=DEFAULT_POW,
                         metavar='POWER_LAW', type=float, dest="p",
                         help='The power law for distance weighting of points in the density ' \
                              'calculation, i.e. the p in 1/dist^p. Default: %.2f' % DEFAULT_POW)
                              
  arg_parse.add_argument('-cache', '--cache_result_dir', metavar='DIR_NAME', default=None, dest='cache',
                         help='If set, saves intermediate results to the specified directory.' \
                              'Makes re-plotting much faster.')

  arg_parse.add_argument('-colors', metavar='COLOR_SCALE', default='w,b,y',
                         help='Optional scale colours as a comma-separated list, e.g. "white,blue,red".' \
                              'or colormap (scheme) name, as used by matplotlib. ' \
                              'Note: #RGB style hex colours must be quoted e.g. "#FF0000,#0000FF" ' \
                              'See: %s This option overrides -b.' % COLORMAP_URL)
                         
  args = vars(arg_parse.parse_args(argv))
                                
  struc_paths1 = args['i']
  struc_paths2 = args['s'] or []
  data_tracks = args['d']
  data_labels = args['l']
  screen_gfx = args['g']
  min_sep = args['m']
  out_path = args['o']
  radius = args['r']
  power = args['p']
  cmap = args['colors']
  cache_dir = args['cache']
   
  if not struc_paths1:
    arg_parse.print_help()
    sys.exit(1)  

  if cmap:
    if ',' in cmap:
      colors = cmap.split(',')
      try:
        cmap = LinearSegmentedColormap.from_list(name='pcm', colors=colors, N=255)    
      except ValueError as err:
        util.warn(err)
        util.critical('Invalid colour specification')
      
    else:
      try:
        cmap = plt.get_cmap(cmap)
      except ValueError as err:
        util.warn(err)
        util.critical('Invalid colourmap name. See: %s' % COLORMAP_URL)
  
  if out_path and screen_gfx:
    util.warn('Output file will not be written in screen graphics (-g) mode')
    out_path = None
  
  for in_path in struc_paths1 + struc_paths2 + data_tracks:
    if not os.path.exists(in_path):
      util.critical('Input file "{}" could not be found'.format(in_path))
  
  structure_data_density(struc_paths1, struc_paths2,
                         data_tracks, data_labels,
                         out_path, screen_gfx,
                         radius, min_sep, power,
                         cmap, cache_dir)
  
if __name__ == "__main__":
  sys.path.append(dirname(dirname(__file__)))
  main()


"""
TTD
---
Bootstrap graph errors

Density matrices
+ all structures vs null (set hier order)

+ g1 vs null, g2 vs null (same order as above)
  g1 vs g2 (SJ), g1-g2

Labels, dendrogram on colour matrices
+ Split between struc groups

Structure differences matrix 
+ hierarchical order
+ cols from data pairs
+ separately for two groups

--

Proper distrib p-values
MD5sum caching 
Reduce memory use

--


New programs

 Compare tracks
 - Correlate region binned data track values
 - Quantile or log norm
 - Different tracks relative to a reference
   
   + Reference is x-axis
   + Y-axis distribution of others
     Separate sub-plots
     - Density plot
     - Box/violin plot
     - Scatter
     Combined plot
     - Lines with errs

 - Measure sequential overlap
 
 Combine tracks
 
 Structure-derived tracks
 
./nuc_tools structure_data_density -h  

./nuc_tools structure_data_density /home/tjs23/gh/nuc_tools/n3d/Cell[12]_100kb_x10.n3d -d /data/bed/H3K4me3_hap_EDL.bed /data/bed/H3K27me3_hap_EDL.bed /data/bed/H3K9me3_hap_EDL.bed -cache sdd_temp

./nuc_tools structure_data_density /home/tjs23/gh/nuc_tools/n3d/Cell[12]_100kb_x10.n3d -d /data/bed/H3K4me3_hap_EDL.bed /data/bed/H3K27me3_hap_EDL.bed /data/bed/H3K9me3_hap_EDL.bed /data/bed/Oct4_GEO.bed /data/bed/p300_GEO.bed /data/bed/H3K36me3_hap_EDL.bed -l H3K4me3 H3K27me3 H3K9me3 Oct4 p300 H3K36me3 H3K27ac -cache sdd_temp

./nuc_tools structure_data_density /home/tjs23/gh/nuc_tools/n3d/Cell1_100kb_x10.n3d -s /home/tjs23/gh/nuc_tools/n3d/Cell2_100kb_x10.n3d -d /data/bed/H3K4me3_hap_EDL.bed /data/bed/H3K27me3_hap_EDL.bed /data/bed/H3K9me3_hap_EDL.bed /data/bed/Oct4_GEO.bed /data/bed/p300_GEO.bed /data/bed/H3K36me3_hap_EDL.bed -l H3K4me3 H3K27me3 H3K9me3 Oct4 p300 H3K36me3 H3K27ac -cache sdd_temp
"""
