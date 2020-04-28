import os, sys, math, multiprocessing, ctypes, subprocess
import numpy as np
from time import time
from collections import defaultdict
from glob import glob

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
DEFAULT_PDF_OUT = 'sdd_out_job{}_S{}-D{}.pdf'
DEFAULT_MAX_RADIUS = 5.0
DEFAULT_POW = 3.0
MIN_FLOAT = sys.float_info.min
DENSITY_KEY = '_DENSITY_'
PDF_DPI = 200
NUM_BOOTSTRAP = 100
DEFAULT_OFFSET_NULL = 5

DENS_OBS = 'observed data'
DENS_ALL = 'all sites'
DENS_SHU = 'shuffled sites'
DENS_OFF = 'offset sites'
DENS_USR = 'user site'
DENS_KEYS = (DENS_OBS, DENS_ALL, DENS_SHU, DENS_OFF, DENS_USR)

MAX_CPU = multiprocessing.cpu_count()


def _run_dens_calc(d_mat, vals, num_cpu=MAX_CPU):

  from core.nuc_parallel import run
  
  weights = vals[None,:] # Size 1 first dim for broadcasting
  
  def _job(idx):
    a, b = idx
    return (d_mat[a:b] * weights).sum(axis=1) # For each test site multiply individual local density (wrt all data sites) by unversal data site values 
  
  n, m = d_mat.shape
  
  split_idx = np.linspace(0, n, num_cpu+1).astype(int)

  job_data = [(split_idx[i], split_idx[i+1]) for i in range(num_cpu)]
  
  results = run(_job, job_data, num_cpu=num_cpu, verbose=False)
  
  return np.concatenate(results)
        
        
def get_data_density(d_mat, chromo_limits, data_bed_path, bin_size, num_cpu):

  from formats import bed
  from nuc_tools import util
    
  data_regions, data_values = bed.load_bed_data_track(data_bed_path)[:2]
  chromos = sorted(chromo_limits)
      
  # Get binned data track values over universal/max extent chromosome
  
  data_hists = {}
  
  # Structures can have a different regions
  
  for chromo in chromo_limits:
    start, end = chromo_limits[chromo]
    data_hists[chromo] = util.bin_region_values(data_regions[chromo],
                                                data_values[chromo],
                                                bin_size, start, end)
  # Get flat arrays for all particles from separate chromosomes

  data_idx = []
  data_values = []
  
  # Get flat arrays of track data, using same regions as particle arrays
  a = 0
  for chromo in chromos:    
    hist = data_hists[chromo]
    idx  = hist.nonzero()[0]
    data_values.append( hist[idx] )
    data_idx.append(a + idx)
    a += len(hist)
    start, end = chromo_limits[chromo]                             

  data_values = np.concatenate(data_values, axis=0)
  data_idx = np.concatenate(data_idx, axis=0)

  densities = _run_dens_calc(d_mat[:,data_idx], data_values, num_cpu) # densities at all sites
  densities /= densities.sum()
  
  return densities
  
  
def get_point_density(d_mat, anch_idx, off_idx, user_idx, data_idx, data_values, num_cpu=MAX_CPU):
  
  from core.nuc_parallel import run
  
  # First dim is obs or null sites
  # Second dim is data sites
 
  #t0 = time()
 
  na, nb = d_mat.shape
  
  d_mat = d_mat[:,data_idx]  # Only where test data is present
  
  densities = _run_dens_calc(d_mat, data_values, num_cpu)
 
  dens_dict = {}
  
  # A points vs B points & values
  dens_dict[DENS_OBS] = densities[anch_idx]
  
  # All points vs B points & values
  dens_dict[DENS_ALL] = densities
  
  # A points vs B points with shuffled B values
  #dens_dict[DENS_SHU] = (d_mat[anch_idx] * w_mat2).sum(axis=1)
  #dens_dict[DENS_SHU] = _run_dens_calc(d_mat[anch_idx], shuff_data_values)
  
  # Offset (from A) points vs B  
  dens_dict[DENS_OFF] = densities[off_idx]
  
  # User specified points vs B
  if len(user_idx):
    dens_dict[DENS_USR] = densities[user_idx]
  
  #print 'TT: %.3f' % (time()-t0)
  
  return dens_dict


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
    end   = particle_size * int(pos[-1]/particle_size)
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


def get_pde(dens_mat, chromo_limits, anchor_bed_path, density_bed_path,
            bin_size, null_bed, null_offset, num_cpu):
  """
  """
  from formats import n3d, bed
  from nuc_tools import util
  from matplotlib import pyplot as plt
  
  # # # particle_size must be checked to be same in all structures
  # # # null hypothesis could include a max seq sep 
  
  anch_regions, anch_values = bed.load_bed_data_track(anchor_bed_path)[:2]
  
  if anchor_bed_path == density_bed_path:
    data_regions, data_values = anch_regions, anch_values
  else:
    data_regions, data_values = bed.load_bed_data_track(density_bed_path)[:2]
  
  if null_bed:
    null_regions, null_values = bed.load_bed_data_track(null_bed)[:2]
  else:
    null_regions = None
  
  chromos = sorted(chromo_limits)
      
  # Get binned data track values over universal/max extent chromosome
  
  data_hists = {}
  anch_hists = {}
  null_hists = {}
  
  # Structures can have a different regions
  
  for chromo in chromo_limits:
    start, end = chromo_limits[chromo]
    data_hists[chromo] = util.bin_region_values(data_regions[chromo], data_values[chromo],
                                           bin_size, start, end)
    
    hist = data_values[chromo]
    anch_hists[chromo] = util.bin_region_values(anch_regions[chromo], anch_values[chromo],
                                           bin_size, start, end)
    
    if null_regions:
      null_hists[chromo] = util.bin_region_values(null_regions[chromo], np.ones(len(null_regions[chromo])),
                                             bin_size, start, end)
                               
  # Get flat arrays for all particles from separate chromosomes

  user_idx = []
  anch_idx = []
  data_idx = []
  data_values = []
  #anch_values = []
  
  # Get flat arrays of track data, using same regions as particle arrays
  a = 0
  b = 0
  c = 0
  for chromo in chromos:    
    hist = data_hists[chromo]
    idx  = hist.nonzero()[0]
    data_values.append( hist[idx] )
    data_idx.append(a + idx)
    a += len(hist)
    
    hist2 = anch_hists[chromo]
    idx2 = hist2.nonzero()[0]
    anch_idx.append(b +  idx2)
    b += len(hist2)
    
    if null_regions:
      hist3 = null_hists[chromo]
      idx3 = hist3.nonzero()[0]
      user_idx.append(c +  idx3)
      c += len(hist3)
    
  data_values = np.concatenate(data_values, axis=0)
  anch_idx = np.concatenate(anch_idx, axis=0)
  data_idx = np.concatenate(data_idx, axis=0)
  
  #shuff_data_values = data_values[:]
  #np.random.shuffle(shuff_data_values)

  dens_dict = {}
  if null_regions:
    user_idx = np.concatenate(user_idx, axis=0)

  off_idx = np.clip(np.unique(np.concatenate([anch_idx+null_offset, anch_idx-null_offset, anch_idx+null_offset-1, anch_idx-null_offset+1], axis=0)), 0, len(dens_mat)-1)
  dens_dict = get_point_density(dens_mat, anch_idx, off_idx, user_idx, data_idx, data_values, num_cpu)      

  return dens_dict
    
    
def overlap_plot(density_dict, data_labels, pdf=None, cmap='Blues', split_idx=None, is_primary=True, n_bins=100):
 
  n_tracks = len(data_labels)
  plot_size = max(8, n_tracks)
  fig, axarr = plt.subplots(n_tracks, n_tracks) # , sharex=True, sharey=True)
  fig.set_size_inches(plot_size, plot_size)
  
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.05)
  
  overlap_mat = np.zeros((n_tracks, n_tracks))
  
  if split_idx is None:
    plt.suptitle('Density co-localization')
    
  elif split_idx and is_primary:
    plt.suptitle('Density co-localization : primary structures')
    
  else:
    plt.suptitle('Density co-localization : secondary structures')
  
  for row in range(n_tracks):
    for col in range(n_tracks):
      if n_tracks > 1:
        ax = axarr[row, col]
      else:
        ax = axarr
 
      if split_idx is None:
        dens_a = np.concatenate(density_dict[row])
        dens_b = np.concatenate(density_dict[col])
 
      elif is_primary:
        dens_a = np.concatenate(density_dict[row][:split_idx])
        dens_b = np.concatenate(density_dict[col][:split_idx])
 
      else:
        dens_a = np.concatenate(density_dict[row][split_idx:])
        dens_b = np.concatenate(density_dict[col][split_idx:])
      
      nz = (dens_a > 0) & (dens_b > 0)
      
      ax.scatter(np.log10(dens_a[nz]), np.log10(dens_b[nz]), s=2, alpha=0.3)

      ax.hist2d(x_vals, y_vals,
              bins=hist_bins2d, range=(hist_range, hist_range),
              cmap=cmap)      
      #coloc = np.log10(1.0 + dens_a[nzb])

      #ax.hist(coloc, bins=n_bins, normed=True)
      
      #ax.text(0.05, 0.95, '$\\rho$={:.3f}\n$n$={:,}'.format(r,len(x_vals)), transform=ax.transAxes,
      #        color='#404040', verticalalignment='top', alpha=0.5, fontsize=8)

      if row == 0:
        axr = ax.twiny()
        axr.set_xticks([])
        axr.set_xlabel(data_labels[col], fontsize=11)

      if col == n_tracks-1:
        axr = ax.twinx()
        axr.set_yticks([])
        axr.set_ylabel(data_labels[row], fontsize=11)

      if (row == n_tracks-1) and (col == 0):
        ax.set_xlabel('Overlap', fontsize=11)

      if (col == 0) and (row == 0):
        ax.set_ylabel('Probability density', fontsize=11)

  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show()
    
  plt.close()
    
def correlation_plot(dens_exp, data_labels, pdf=None, cmap='Blues', split_idx=None, is_primary=True, max_dens=4.5, hist_bins2d=50):
 
  n_tracks = len(data_labels)
  hist_range = (0.0, max_dens)
  plot_size = max(8, n_tracks)
  fig, axarr = plt.subplots(n_tracks, n_tracks, sharex=True, sharey=True)
  fig.set_size_inches(plot_size, plot_size)
  
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

    if (row == n_tracks-1) and (col == 0):
      ax.set_xlabel('Density', fontsize=11)

    if (col == 0) and (row == 0):
      ax.set_ylabel('Density', fontsize=11)
  
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show()
    
  plt.close()


def density_plot(title, mat, val_label, data_labels, cmap, pdf, vmin=None, vmax=None):  
  
  pw = 0.65
  bot = 0.15
  left = 0.15
  
  n_tracks = len(mat)
  
  if vmax is None:
    vmax = mat.max()
  
  if vmin is None:
    if mat.min() < 0:
      vmin = min(mat.min(), -vmax)
    else:
      vmin = 0.0
      
  plot_size = max(8, 0.25*n_tracks)
  fig = plt.figure()    
  fig.set_size_inches(plot_size, plot_size)
  
  plt.suptitle(title)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.04, hspace=0.04)
  
  ax1 = fig.add_axes([left, bot, pw, pw])
  
  dist_mat = distance.pdist(mat)
  linkage = hierarchy.linkage(dist_mat, method='ward', optimal_ordering=True)
  order = hierarchy.leaves_list(linkage)[::-1]
  
  xylabels = [data_labels[i] for i in order]
  xylabel_pos = np.linspace(0.0, n_tracks-1, n_tracks)
  
  cax1 = ax1.matshow(mat[order][:,order], cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
  ax1.set_xticklabels(xylabels, fontsize=8, rotation=90.0)
  ax1.set_yticklabels(xylabels, fontsize=8)

  ax1.xaxis.set_ticks(xylabel_pos)
  ax1.xaxis.set_tick_params(which='both', direction='out')
 
  ax1.yaxis.set_ticks(xylabel_pos)
  ax1.yaxis.set_tick_params(which='both', direction='out')

  dgax1 = fig.add_axes([pw+left, bot, bot, pw]) # left, bottom, w, h  
  ddict = hierarchy.dendrogram(linkage, orientation='right', labels=data_labels,
                               above_threshold_color='#000000', no_labels=True,
                               link_color_func=lambda k: '#000000', ax=dgax1)
  dgax1.set_xticklabels([])
  dgax1.set_xticks([])
  dgax1.set_axis_off()               
  
  cbax1 = fig.add_axes([left, bot-0.05, pw, 0.02]) # left, bottom, w, h
  
  cbar = plt.colorbar(cax1, cax=cbax1, orientation='horizontal')
  cbar.ax.tick_params(labelsize=8)
  cbar.set_label(val_label, fontsize=9)
  
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show() 
  
  plt.close()
  
  return order
  

def comparison_density_plot(title, mat1, mat2, mat12, order, val_label, data_labels, cmap, cmap2, pdf, vmin=None, vmax=None):  
  
  sz = 0.33
  bot = 0.08
  left = 0.1
  mid = bot+sz+left+0.02
  
  n_tracks = len(mat1)
  
  if vmax is None:
    vmax = max(mat1.max(), mat2.max())
  
  if vmin is None:
    if mat1.min() < 0:
      vmin = min(mat1.min(), mat2.min(), -vmax)
    else:
      vmin = 0.0
      
  plot_size = max(8, 0.25*n_tracks)
  fig = plt.figure()    
  fig.set_size_inches(plot_size, plot_size)
  
  plt.suptitle(title)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.04, hspace=0.04)
  
  xylabels = [data_labels[i] for i in order]
  xylabel_pos = np.linspace(0.0, n_tracks-1, n_tracks)
  
  ax1 = fig.add_axes([left, mid, sz, sz])
  ax2 = fig.add_axes([left+sz+left, mid, sz, sz])
  ax4 = fig.add_axes([left+sz+left, bot, sz, sz])
    
  ax1.set_xlabel('Group 1')
  ax2.set_xlabel('Group 2')
  ax4.set_xlabel('Difference')

  m1 = mat1[order][:,order]
  m2 = mat2[order][:,order]
  cax1 = ax1.matshow(m1, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
  cax2 = ax2.matshow(m2, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
  
  dmax = 0.5 * vmax  
  cax4 = ax4.matshow(m1-m2, cmap=cmap2, aspect='auto',  vmin=-dmax, vmax=dmax)  

  if mat12 is not None:
    ax3 = fig.add_axes([left, bot, sz, sz])
    ax3.set_xlabel('Group 1 vs Group 2')
    m12 = mat12[order][:,order]
    cax3 = ax3.matshow(m12, cmap=cmap, aspect='auto')
    axarr = [ax1, ax2, ax3, ax4]
  else:
    axarr = [ax1, ax2, ax4]
   
  for ax in axarr:
    ax.set_xticklabels(xylabels, fontsize=8, rotation=45.0)
    ax.set_yticklabels(xylabels, fontsize=8)

    ax.xaxis.set_ticks(xylabel_pos)
    ax.xaxis.set_tick_params(which='both', direction='out')
 
    ax.yaxis.set_ticks(xylabel_pos)
    ax.yaxis.set_tick_params(which='both', direction='out')

  cbax1 = fig.add_axes([left+left+sz+sz+0.02, mid, 0.02, sz]) # left, bottom, w, h
  cbax2 = fig.add_axes([left+left+sz+sz+0.02, bot, 0.02, sz]) # left, bottom, w, h
  
  cbar1 = plt.colorbar(cax1, cax=cbax1, orientation='vertical')
  cbar1.ax.tick_params(labelsize=8)
  cbar1.set_label(val_label, fontsize=9)

  cbar2 = plt.colorbar(cax4, cax=cbax2, orientation='vertical')
  cbar2.ax.tick_params(labelsize=8)
  cbar2.set_label('Difference', fontsize=9)
  
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show() 
  
  plt.close()


def _dkl(obs, exp):
   nz = (obs > 0) & (exp > 0)
   obs = obs.astype(float)/float(obs.sum())
   exp = exp.astype(float)/float(exp.sum())
   return (obs[nz] * np.log(obs[nz]/exp[nz])).sum()


def calc_enrichments(n_tracks, struc_paths1, struc_paths2, dens_obs, dens_all, dens_null,
                     hist_bins, symm_mat=True, hist_range=(0.0, 100.0), n_bootstrap=NUM_BOOTSTRAP):
  
  def _corr(vals_a, vals_b):
    nz = (vals_a > 0) & (vals_b > 0)
    vals_a = np.log10(vals_a[nz])
    vals_b = np.log10(vals_b[nz])
    r, p = stats.pearsonr(vals_a, vals_b)
    return r
    
  hist_null, hist_obs0, hist_obs1, hist_obs2 = {}, {}, {}, {}
  err0, err1, err2 = {}, {}, {}
  
  js_mat_all = np.zeros((n_tracks, n_tracks))
  corr_mat_all = np.zeros((n_tracks, n_tracks))
  
  if struc_paths2:
    js_mat1 = np.zeros((n_tracks, n_tracks))
    js_mat2 = np.zeros((n_tracks, n_tracks))
    js_mat12 = np.zeros((n_tracks, n_tracks))
    corr_mat1 = np.zeros((n_tracks, n_tracks))
    corr_mat2 = np.zeros((n_tracks, n_tracks))
  else:
    js_mat1 = js_mat_all
    js_mat2 = None
    js_mat12 = None
    corr_mat1 = None
    corr_mat2 = None
 
  a = len(struc_paths1)
    
  for key in dens_null:
    i, j = key 
    
    obs0 = np.concatenate(dens_obs[key])
    exp0 = np.concatenate(dens_null[key])
    bg0 = np.sort(np.concatenate(dens_all[key]))
    n = float(len(bg0))
    m = len(obs0)
     
    obs0 = np.searchsorted(bg0, obs0).astype(float)
    obs0 *= 100.0/(n+1.0)

    hist_obs0[key], edges = np.histogram(obs0, bins=hist_bins, range=hist_range)    

    exp0 = np.searchsorted(bg0, exp0).astype(float)
    exp0 *= 100.0/(n+1.0)
 
    null, edges = np.histogram(exp0, bins=hist_bins, range=hist_range)
    hist_null[key], edges = np.histogram(exp0, bins=hist_bins, range=hist_range)
    
    bs_hists = np.zeros((n_bootstrap, hist_bins))
    for k in range(n_bootstrap):
      idx = np.random.randint(0, m-1, m)
      sample_wr = obs0[idx]
      bs_hists[k], edges = np.histogram(sample_wr, bins=hist_bins, range=hist_range)
    
    err0[key] = bs_hists.std(axis=0)
    
    if struc_paths2: # Compare two structure groups
      bg1 = np.sort(np.concatenate(dens_null[key][:a]))
      obs1 = np.concatenate(dens_obs[key][:a])
      obs1 = np.searchsorted(bg1, obs1).astype(float)
      n = float(len(bg1))      
      obs1 *= 100.0/(n+1.0)
      m1 = len(obs1)
      
      bg2 = np.sort(np.concatenate(dens_null[key][a:]))
      obs2 = np.concatenate(dens_obs[key][a:])
      obs2 = np.searchsorted(bg2, obs2).astype(float)
      n = float(len(bg2))      
      obs2 *= 100.0/(n+1.0)
      m2 = len(obs2)
      
      hist_obs1[key], edges = np.histogram(obs1, bins=hist_bins, range=hist_range)
      hist_obs2[key], edges = np.histogram(obs2, bins=hist_bins, range=hist_range)
      
      bs_hists1 = np.zeros((n_bootstrap, hist_bins))
      bs_hists2 = np.zeros((n_bootstrap, hist_bins))
      for k in range(n_bootstrap):
        idx = np.random.randint(0, m1-1, m1)
        bs_hists1[k], edges = np.histogram(obs1[idx], bins=hist_bins, range=hist_range)
        idx = np.random.randint(0, m2-1, m2)
        bs_hists2[k], edges = np.histogram(obs2[idx], bins=hist_bins, range=hist_range)
      
      err1[key] = bs_hists1.std(axis=0)
      err2[key] = bs_hists2.std(axis=0)
      
    else: # Compare vs random
      hist_obs1[key] = hist_obs0[key]

    js_mat_all[i,j] = _dkl(hist_obs0[key], null)
    corr_mat_all[i,j] = _corr(np.concatenate(dens_null[(i,i)]), np.concatenate(dens_null[key]))
    
    if struc_paths2:
      js_mat1[i,j] = _dkl(hist_obs1[key], null)
      js_mat2[i,j] = _dkl(hist_obs2[key], null)
      js_mat12[i,j] = _dkl(hist_obs1[key], hist_obs2[key])
      corr_mat1[i,j] = _corr(np.concatenate(dens_null[(i,i)][:a]), np.concatenate(dens_null[key][:a]))
      corr_mat2[i,j] = _corr(np.concatenate(dens_null[(i,i)][a:]), np.concatenate(dens_null[key][a:]))

  if symm_mat:
    js_mat_all += js_mat_all.T
    js_mat_all /= 2.0
    
    if struc_paths2:
      js_mat1 +=js_mat1.T
      js_mat2 +=js_mat2.T
      js_mat12 +=js_mat12.T
      js_mat1 /= 2.0
      js_mat2 /= 2.0
      js_mat12 /= 2.0
  
  histograms = (hist_null, hist_obs0, hist_obs1, hist_obs2)
  div_mats = (js_mat_all, js_mat1, js_mat2, js_mat12)
  corr_mats = (corr_mat_all, corr_mat1, corr_mat2)
  errors = (err0, err1, err2)
  
  return histograms, div_mats, corr_mats, errors
  
  
def plot_enrichment_distribs(title, data_labels, histogrms, errors, hist_bins, pdf):
  
  hist_null, hist_obs0, hist_obs1, hist_obs2 = histogrms
  err0, err1, err2 = errors
  n_tracks = len(data_labels)
  
  plot_width = max(10, 1.25*n_tracks)
  plot_height = max(8, n_tracks)
  fig, axarr = plt.subplots(n_tracks, n_tracks, sharey=True, sharex=True)    
  fig.set_size_inches(plot_width, plot_height)
  
  plt.suptitle(title)
  plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.9, wspace=0.05, hspace=0.05)
  
  hist_range = (0.0, 100.0)
  exp_val = 1.0/hist_bins
  
  for key in hist_null:
    row, col = key
    
    if n_tracks > 1:
      ax = axarr[row, col]
    else:
      ax = axarr
    
    if hist_obs2: # Compare two structure groups
      hist_obs = hist_obs1[key]
      hist_exp = hist_obs2[key]
      label1 = 'Group1'
      label2 = 'Group2'
      
    else: # Compare vs random
      err1 = err0
      hist_obs = hist_obs0[key]
      hist_exp = hist_null[key]
      label1 = 'Obs'
    
    null = hist_null[key]
    n_obs = hist_obs.sum()
    x_vals = np.linspace(0.0, 100.0, hist_bins) 
    
    nz = (hist_obs > 0) & (hist_exp > 0)

    hist_obs = hist_obs[nz].astype(float)
    hist_exp = hist_exp[nz].astype(float)
    null = null[nz].astype(float)
    
    z, pv = stats.power_divergence(hist_obs, hist_exp*n_obs/hist_exp.sum(), lambda_=0)
    pv = max(pv, MIN_FLOAT)
    
    x_vals = x_vals[nz]
    
    s1 = float(hist_obs.sum())
    s2 = float(hist_exp.sum())
    
    hist_obs /= s1
    hist_exp /= s2
    null /= float(null.sum())
       
    ax.plot(x_vals, hist_obs, color='#0080FF', label=label1)
    #ax.errorbar(x_vals, hist_obs, err1[key]/s1, color='#0080FF', label=label1)
    
    if hist_obs2:
      ax.plot(x_vals, hist_exp, color='#FF4000', label=label2)
      #ax.errorbar(x_vals, hist_exp, err2[key]/s2, color='#808080', label=label2)
    
    ax.plot(x_vals, null, color='#808080', alpha=0.5, label='Background')
    
    #ax.plot((0.0, 100.0), (exp_val, exp_val), color='#808080', ls='--', alpha=0.5, label='Exp')
    
    js = 0.5 * (hist_obs * np.log(hist_obs/hist_exp)).sum()
    js += 0.5 * (hist_exp * np.log(hist_exp/hist_obs)).sum()
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

    if (row == n_tracks-1) and (col == 0):
      ax.set_xlabel('Density percentile', fontsize=11)
    
    if col == n_tracks-1:
      axr = ax.twinx()
      axr.set_yticks([])
      axr.set_ylabel('{}\nn={:,}'.format(data_labels[row], n_obs), fontsize=8)
     
    if (col == 0) and (row == 0):
      ax.set_ylabel('Fraction total', fontsize=11)
  
  ax = fig.add_axes([0.2, 0.9, 0.1, 0.1])
  ax.plot([], [], color='#0080FF', label=label1)
    
  if hist_obs2:
    ax.plot([], [], color='#808080', label=label2)
    
  ax.plot([], [], color='#808080', alpha=0.5, label='Background')
  ax.axis('off')
  ax.legend(fontsize=8, frameon=False, ncol=2)
    
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show() 
  
  plt.close()

def plot_separate_structures(title, data_labels, struc_labels, dens_obs, dens_all, dens_exp, split_idx, pdf, cmap, hist_bins, hist_range=(0.0, 100.0)):

  xlabels = []
  keys = []
  for i, label1 in enumerate(data_labels):
    for j, label2 in enumerate(data_labels[i:], i):
      xlabels.append('%s\n%s' % (label1, label2))
      keys.append((i,j))
      
  n_tracks = len(data_labels)
  n_strucs = len(struc_labels)
  n_cols = len(xlabels)
      
  fig = plt.figure()    
  plot_width = max(8, 0.2*n_cols)
  plot_height =  max(8, 0.2*n_strucs)
  
  fig.set_size_inches(plot_width, plot_height)
  plt.suptitle(title)
  plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.9, wspace=0.05, hspace=0.05)
  
  ax1 = fig.add_axes([0.2, 0.1, 0.5, 0.7])
  
  mat = np.zeros((n_strucs, n_cols))
  null = np.linspace(0.0, 100.0, hist_bins) 
  hist_null, edges = np.histogram(null, bins=hist_bins, range=hist_range)
  is_secondary = []
  
  for row in range(n_strucs):
    for col, key in enumerate(keys):
      i, j = key
      bg1  = np.sort(dens_all[key][row])  # All site normalisation
      obs1 = np.searchsorted(bg1, dens_obs[key][row]).astype(float)  
      obs1 *= 100.0/(float(len(bg1))+1.0)
      exp1 = np.searchsorted(bg1, dens_exp[key][row]).astype(float)  
      exp1 *= 100.0/(float(len(bg1))+1.0)
      
      bg2  = np.sort(dens_all[(j,i)][row]) # All site normalisation
      obs2 = np.searchsorted(bg2, dens_obs[(j,i)][row]).astype(float)     
      obs2 *= 100.0/(float(len(bg2))+1.0)
      exp2 = np.searchsorted(bg2, dens_exp[(j,i)][row]).astype(float)  
      exp2 *= 100.0/(float(len(bg2))+1.0)
      
      hist_obs1, edges = np.histogram(obs1, bins=hist_bins, range=hist_range)
      hist_obs2, edges = np.histogram(obs2, bins=hist_bins, range=hist_range)
      hist_exp1, edges = np.histogram(exp1, bins=hist_bins, range=hist_range)
      hist_exp2, edges = np.histogram(exp2, bins=hist_bins, range=hist_range)

      mat[row, col] =  (_dkl(hist_obs1, hist_exp1) + _dkl(hist_obs2, hist_exp2))/2.0
  
    if row < split_idx:
      is_secondary.append(False)
    else:
      is_secondary.append(True)
      
  dist_mat = distance.pdist(mat)
  linkage_y = hierarchy.linkage(dist_mat, method='ward', optimal_ordering=True)
  order_y = hierarchy.leaves_list(linkage_y)[::-1]

  dist_mat = distance.pdist(mat.T)
  linkage_x = hierarchy.linkage(dist_mat, method='ward', optimal_ordering=True)
  order_x = hierarchy.leaves_list(linkage_x)[::-1]
  
  xlabels = [xlabels[i] for i in order_x]
  xlabel_pos = np.linspace(0.0, n_cols-1, n_cols)
  
  is_secondary = [is_secondary[i] for i in order_y]
  ylabels = [struc_labels[i] for i in order_y]
  ylabel_pos = np.linspace(0.0, n_strucs-1, n_strucs)
  
  cax = ax1.matshow(mat[order_y][:,order_x], cmap=cmap, aspect='auto', vmin=0.0)
  
  ax1.set_xticklabels(xlabels, fontsize=8, rotation=90.0)
  ax1.set_yticklabels(ylabels, fontsize=8)

  ax1.xaxis.set_ticks(xlabel_pos)
  ax1.xaxis.set_tick_params(which='both', direction='out')
 
  ax1.yaxis.set_ticks(ylabel_pos)
  ax1.yaxis.set_tick_params(which='both', direction='out')

  for i, text in enumerate(ax1.get_yticklabels()):
    if is_secondary[i]:
      text.set_color('#E00000')

  ax2 = fig.add_axes([0.7, 0.1, 0.2, 0.7]) # Dendrogram  
  ddict = hierarchy.dendrogram(linkage_y, orientation='right', labels=struc_labels,
                               above_threshold_color='#000000', no_labels=True,
                               link_color_func=lambda k: '#000000', ax=ax2)
  ax2.set_xticklabels([])
  ax2.set_xticks([])
  ax2.set_axis_off()               
  
  ax3 = fig.add_axes([0.2, 0.05, 0.5, 0.02]) # left, bottom, w, h
  
  cbar = plt.colorbar(cax, cax=ax3, orientation='horizontal')
  cbar.ax.tick_params(labelsize=8)
  cbar.set_label('JS divergence', fontsize=9)  
  
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show() 
  
  plt.close()


def _shared_mem_array(arry):

  n, m = arry.shape
  shared = np.ctypeslib.as_array(multiprocessing.Array(ctypes.c_float, n*m).get_obj()).reshape(n, m)
  shared[:,:] = arry
  
  return shared
  
 
def structure_data_density(struc_paths1, struc_paths2, data_tracks, data_labels=None,
                           null_bed=None, null_offset=DEFAULT_OFFSET_NULL,
                           out_path=None, screen_gfx=None, radius=DEFAULT_MAX_RADIUS, 
                           min_sep=DEFAULT_MIN_PARTICLE_SEP, dist_pow=DEFAULT_POW,
                           cmap=plt.get_cmap('Blues'), num_cpu=MAX_CPU, cache_dir=None):
  
  from nuc_tools import io, util
  from formats import n3d
  
  struc_paths1 = sorted(struc_paths1)
  
  if struc_paths2:
    struc_paths2 = sorted(struc_paths2)
    msg = 'Analysing {} data tracks comparing two groups with structures {} and {} respectively.'
    util.info(msg.format(len(data_tracks), len(struc_paths1), len(struc_paths2)))
    n_structs = len(struc_paths1) + len(struc_paths2)
  
  else:
    msg = 'Analysing {} data tracks with {} structures.'
    util.info(msg.format(len(data_tracks), len(struc_paths1)))
    n_structs = len(struc_paths1)
  
  if out_path:
    out_path = io.check_file_ext(out_path, '.pdf')
  
  else:    
    dir_path = dirname(struc_paths1[0])
    
    job = 1
    while glob(os.path.join(dir_path, DEFAULT_PDF_OUT.format(job, '*', '*'))):
      job += 1
    
    file_name = DEFAULT_PDF_OUT.format(job, n_structs, len(data_tracks))
    out_path = os.path.join(dir_path, file_name)

  data_labels = io.check_file_labels(data_labels, data_tracks)
  
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
  dens_dicts = {k:defaultdict(list) for k in DENS_KEYS}
  
  data_density = defaultdict(list)
  
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
    
    """
    for i, data_bed_path in enumerate(data_tracks):
      util.info('  Calculating densities for {}'.format(os.path.basename(data_bed_path)))
      densties = get_data_density(dens_mat, struc_chromo_lims, data_bed_path, bin_size, num_cpu)
      data_density[i].append(densties)
      
    """
    for row, anchor_bed_path in enumerate(data_tracks): # Anchor
      util.info('  Considering {}'.format(os.path.basename(anchor_bed_path)))
      bed_root1 = io.get_file_root(anchor_bed_path)
      
      for col, density_bed_path in enumerate(data_tracks): # Data
        data_key = row, col
        util.info('  .. compared to {}'.format(os.path.basename(density_bed_path)), line_return=True)
        bed_root2 = io.get_file_root(density_bed_path)
 
        pde_dict = get_pde(dens_mat, struc_chromo_lims,
                           anchor_bed_path, density_bed_path,
                           bin_size, null_bed, null_offset, num_cpu)
 
        for typ_key in pde_dict:
          dens_dicts[typ_key][data_key].append(pde_dict[typ_key])
    
  struc_labels = [os.path.basename(os.path.splitext(path)[0]) for path in struc_paths1+struc_paths2]
  
  cmap2 = LinearSegmentedColormap.from_list(name='pcm', colors=['#0060E0','#FFFFFF','#E02000'], N=255)    
  
  hist_bins = 20
  hist_bins2d = 50
  
  #overlap_plot(data_density, data_labels, pdf, cmap, split_idx=None, is_primary=True, n_bins=50)
  
  """
  """
  
  for null in (DENS_ALL, DENS_OFF, DENS_USR): # DENS_SHU
    if not dens_dicts[null]:
      continue
      
    histograms, div_mats, corr_mats, errors = calc_enrichments(n_tracks, struc_paths1, struc_paths2,
                                                               dens_dicts[DENS_OBS], dens_dicts[DENS_ALL],
                                                               dens_dicts[null], hist_bins)
                                                                 
    if null == DENS_ALL:
      # Correlations
      
      if struc_paths2:
        split_idx = len(struc_paths1)
        correlation_plot(dens_dicts[null], data_labels, pdf, cmap, split_idx, True)
        correlation_plot(dens_dicts[null], data_labels, pdf, cmap, split_idx, False)
      else:
        correlation_plot(dens_dicts[null], data_labels, pdf, cmap)
    
 
      order = density_plot('Density correlation - %s' % null, corr_mats[0],
                           'Pearson correlation coefficient', data_labels, cmap, pdf, 0.0, 1.0)
 
      if struc_paths2:
        comparison_density_plot('Density correlation comparison', corr_mats[1], corr_mats[2], None, order,
                                'Pearson correlation coefficient', data_labels, cmap, cmap2, pdf)
    
    # Enrichment distribs
 
    plot_enrichment_distribs('Density enrichments. Background: %s' % null,
                             data_labels, histograms, errors, hist_bins, pdf)

    order = density_plot('Density enrichment - all stuctures. Background: %s' % null, div_mats[0],
                         'Jensen-Shannon divergence', data_labels, cmap, pdf) # , 0.0, 0.1)
 
    if struc_paths2:
      comparison_density_plot('Density enrichment - group comparison. Background: %s' % null,
                               div_mats[1], div_mats[2], div_mats[3], order,
                              'Jensen-Shannon divergence', data_labels, cmap, cmap2, pdf)
    
    plot_separate_structures('Structure density enrichment comparison. Background: %s' % null,
                             data_labels, struc_labels, dens_dicts[DENS_OBS], dens_dicts[DENS_ALL], dens_dicts[null],
                             len(struc_paths1), pdf, cmap, hist_bins) 

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

  arg_parse.add_argument('-m', '--min-particle-sep', default=DEFAULT_MIN_PARTICLE_SEP,
                         metavar='MIN_PARTICLE_SEP', type=int, dest="m",
                         help='The minimum separation of  structure particles for analysis. ' \
                              'Avoids linear sequence correlation effects and distortions due to the ' \
                              'courseness of the 3D model. Default: %d particles' % DEFAULT_MIN_PARTICLE_SEP)

  arg_parse.add_argument('-r', '--max-radius', default=DEFAULT_MAX_RADIUS,
                         metavar='MAX_SERACH_RADIUS', type=float, dest="r",
                         help='The maximum search distance within which data densties are calculated ' \
                              'for a genome structure point. Default: %.2f' % DEFAULT_MAX_RADIUS)

  arg_parse.add_argument('-n', '--num-cpu', default=1, metavar='CPU_COUNT', dest='n',
                         type=int, help='Number of CPU cores to use in parallel  for some parts of the density calculation. ' \
                                        'Default: %d (maximum available)' % MAX_CPU)
                              
  arg_parse.add_argument('-o', '--out-pdf', metavar='OUT_PDF_FILE', default=None, dest='o',
                         help='Optional output PDF file name. If not specified, a default will be used.')

  arg_parse.add_argument('-p', '--power-law', default=DEFAULT_POW,
                         metavar='POWER_LAW', type=float, dest="p",
                         help='The power law for distance weighting of points in the density ' \
                              'calculation, i.e. the p in 1/dist^p. Default: %.2f' % DEFAULT_POW)
                              
  arg_parse.add_argument('-cache', '--cache-result-dir', metavar='DIR_NAME', default=None, dest='cache',
                         help='If set, saves intermediate results to the specified directory.' \
                              'Makes re-plotting much faster.')

  arg_parse.add_argument('-colors', metavar='COLOR_SCALE', default='w,b,y',
                         help='Optional scale colours as a comma-separated list, e.g. "white,blue,red".' \
                              'or colormap (scheme) name, as used by matplotlib. ' \
                              'Note: #RGB style hex colours must be quoted e.g. "#FF0000,#0000FF" ' \
                              'See: %s This option overrides -b.' % util.COLORMAP_URL)

  arg_parse.add_argument('-null', '--null-site-bed', metavar='BED_FILE', default=None, dest='null',
                         help='Optional data track file in BED format to specify comparative regions ' \
                              ' for the calculation of null/background expectation densities. For example' \
                              ' the track may be for all A/euchromatin compartment or accessible sites. ' \
                              'This will augment the all-site, shuffled-site and offset-site null expectations' \
                              ' that are always calculated')
   
  arg_parse.add_argument('-no', '--null-site-offset', default=DEFAULT_OFFSET_NULL,
                         metavar='NULL_SITE_OFFSET', type=int, dest="no",
                         help='The sequence offset to use for calculating a local null/backgound structure' \
                              ' density expectation. Specificied as a number of whole structural particles' \
                              ' Default: %d particles' % DEFAULT_OFFSET_NULL)
                         
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
  null_bed = args['null']
  null_offset = args['no']
  num_cpu = args['n']
   
  if not struc_paths1:
    arg_parse.print_help()
    sys.exit(1)  
  
  if cmap:
    cmap = util.string_to_colormap(cmap)
  
  if out_path and screen_gfx:
    util.warn('Output file will not be written in screen graphics (-g) mode')
    out_path = None
  
  for in_path in struc_paths1 + struc_paths2 + data_tracks:
    io.check_invalid_file(in_path)
  
  nl = len(data_labels)
  nd = len(data_tracks)
  if nl and  nl > nd:
    util.warn('Number of data labels (%d) exceeds the number of data track files (%d)' % (nl, nd))
    data_labels = data_labels[:nd]
  
  structure_data_density(struc_paths1, struc_paths2,
                         data_tracks, data_labels,
                         null_bed, null_offset,
                         out_path, screen_gfx,
                         radius, min_sep, power,
                         cmap, num_cpu, cache_dir)
  
if __name__ == "__main__":
  sys.path.append(dirname(dirname(__file__)))
  main()


"""
TTD
---

COLOUR ACCORDING TO SIGN
Measure fraction of co-density - how much of total is shared 
Proper distrib p-values
MD5sum caching 
--

 Structure-derived tracks
 
./nuc_tools structure_data_density -h  

./nuc_tools structure_data_density /home/tjs23/gh/nuc_tools/n3d/Cell[12]_100kb_x10.n3d -d /data/bed/H3K4me3_hap_EDL.bed /data/bed/H3K27me3_hap_EDL.bed /data/bed/H3K9me3_hap_EDL.bed  /data/bed/H3K27ac_GEO.bed  -l H3K4me3 H3K27me3 H3K9me3 H3K27ac -cache sdd_temp2

./nuc_tools structure_data_density /home/tjs23/gh/nuc_tools/n3d/Cell[12]_100kb_x10.n3d -d /data/bed/H3K4me3_hap_EDL.bed /data/bed/H3K27me3_hap_EDL.bed /data/bed/H3K9me3_hap_EDL.bed /data/bed/Oct4_GEO.bed /data/bed/p300_GEO.bed /data/bed/H3K36me3_hap_EDL.bed -l H3K4me3 H3K27me3 H3K9me3 Oct4 p300 H3K36me3 H3K27ac -cache sdd_temp

./nuc_tools structure_data_density /home/tjs23/gh/nuc_tools/n3d/Cell[23]_100kb_x10.n3d -s /home/tjs23/gh/nuc_tools/n3d/Cell[14]_100kb_x10.n3d -d /data/bed/H3K4me3_hap_EDL.bed /data/bed/H3K27me3_hap_EDL.bed /data/bed/H3K9me3_hap_EDL.bed /data/bed/Oct4_GEO.bed /data/bed/p300_GEO.bed /data/bed/H3K36me3_hap_EDL.bed -l H3K4me3 H3K27me3 H3K9me3 Oct4 p300 H3K36me3 -cache sdd_temp

./nuc_tools structure_data_density /home/tjs23/gh/nuc_tools_bak/n3d/Cell[2358]_100kb_x10.n3d -s /home/tjs23/gh/nuc_tools_bak/n3d/Cell[1467]_100kb_x10.n3d -null /data/bed/A_comp.bed -d /data/bed/H3K4me3_hap_EDL.bed /data/bed/H3K27me3_hap_EDL.bed /data/bed/H3K9me3_hap_EDL.bed /data/bed/H3K27ac_GEO.bed -l H3K4me3 H3K27me3 H3K9me3 H3K27ac -o /home/tjs23/Desktop/test_sdd.pdf

"""
