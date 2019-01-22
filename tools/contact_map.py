import datetime
import numpy as np
import os, sys

from math import ceil
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator

PROG_NAME = 'contact_map'
VERSION = '1.0.0'
DESCRIPTION = 'Chromatin contact (NCC or NPZ format) Hi-C contact map PDF display module'
DEFAULT_CIS_BIN_KB = 250
DEFAULT_TRANS_BIN_KB = 500
DEFAULT_MAIN_BIN_KB = 1000
DEFAULT_SC_MAIN_BIN_KB = 5000
DEFAULT_SC_CHR_BIN_KB = 500
DEFAULT_SMALLEST_CONTIG = 0.1

import warnings
warnings.filterwarnings("ignore")

def _downsample_matrix(in_array, new_shape, as_mean=False):
    
  p, q = in_array.shape
  n, m = new_shape
  
  if (p,q) == (n,m):
    return in_array
  
  if p % n == 0:
    pad_a = 0
  else:
    pad_a = n * int(1+p//n) - p

  if q % m == 0:
    pad_b = 0
  else:
    pad_b = m * int(1+q//m) - q 
  
  if pad_a or pad_b:
    in_array = np.pad(in_array, [(0,pad_a), (0,pad_b)], 'constant')
    p, q = in_array.shape
      
  shape = (n, p // n,
           m, q // m)
  
  if as_mean:
    return in_array.reshape(shape).mean(-1).mean(1)
  else:
    return in_array.reshape(shape).sum(-1).sum(1)

def _get_chromo_offsets(bin_size, chromos, chromo_limits):
  
  chromo_offsets = {}
  label_pos = []
  n = 0
  for chromo in chromos: # In display order
    s, e = chromo_limits[chromo]
    c_bins = int(ceil(e/float(bin_size))) - int(s/bin_size)
    chromo_offsets[chromo] = s, n, c_bins # Start bp, start bin index, num_bins
    label_pos.append(n + c_bins/2)
    n += c_bins
    #n += 1 # Add space between chromos on matrix
  
  return n, chromo_offsets, label_pos


def _get_cis_expectation(obs):

  n = len(obs)
  sobs = obs.sum()
  sep_dict = defaultdict(list) 
 
  for d in range(1, n):
    idx1 = np.array(range(n-d))
    idx2 = idx1 + d
    idx = (idx1, idx2)
    sep_dict[d] = obs[idx]
 
  sep_sig = np.zeros(n, float)
 
  for i in range(n):
    if i in sep_dict:
      sep_sig[i] = np.mean(sep_dict[i])

  expt = np.zeros((n, n), float)

  for i in range(n):
    expt[i,:i] = sep_sig[:i][::-1]
    expt[i,i:] = sep_sig[:n-i]
  
  vals = obs.sum(axis=0).astype(float)
  vals /= vals.sum()
 
  expt *= np.outer(vals, vals)
  expt *= sobs/expt.sum()
  
  return expt

  
def get_corr_mat(obs, clip=5.0):

  obs -= np.diag(np.diag(obs))
  expt = _get_cis_expectation(obs)

  prod = expt * obs
  nz = prod != 0.0
 
  log_ratio = obs.copy()
  log_ratio[nz] /= expt[nz]
  log_ratio[nz] = np.log(log_ratio[nz])
  log_ratio = np.clip(log_ratio, -clip, clip)
 
  corr_mat = np.corrcoef(log_ratio)
  corr_mat -= np.diag(np.diag(corr_mat))

  np.nan_to_num(corr_mat, copy=False)

  
  return corr_mat


def get_trans_corr_mat(obs_a, obs_b, obs_ab, clip=5.0):
  
  n = len(obs_a)
  m = len(obs_b)
  z = n+m

  obs_a -= np.diag(np.diag(obs_a))
  obs_b -= np.diag(np.diag(obs_b))
  mat = np.zeros((z, z))
  
  # Add cis A
  
  expt = _get_cis_expectation(obs_a)
  nz = (expt * obs_a) != 0.0  
  log_ratio = obs_a.copy()
  log_ratio[nz] /= expt[nz]
  
  mat[:n,:n] = log_ratio
  
  # Add cis B
  
  expt = _get_cis_expectation(obs_b)
  nz = (expt * obs_b) != 0.0  
  log_ratio = obs_b.copy()
  log_ratio[nz] /= expt[nz]
  
  mat[n:,n:] = log_ratio
  
  # Add trans
  
  expt = np.full((n,m), obs_ab.mean())
  sobs = obs_ab.sum()
  
  vals_a = obs_a.sum(axis=0) + obs_ab.sum(axis=1)
  vals_b = obs_b.sum(axis=0) + obs_ab.sum(axis=0)
 
  expt *= np.outer(vals_a, vals_b)
  expt *= sobs/expt.sum()
  
  nz = (expt * obs_ab) != 0.0  
  log_ratio = obs_ab.copy()
  log_ratio[nz] /= expt[nz]
  
  mat[:n,n:] = log_ratio
  mat[n:,:n] = log_ratio.T
  
  nz = mat.nonzero()
  mat[nz] = np.log(mat[nz])
  mat = np.clip(mat, -clip, clip)
  
  corr_mat = np.corrcoef(mat)
    
  corr_mat = corr_mat[:n,n:]
    
  np.nan_to_num(corr_mat, copy=False)
  
  return corr_mat


def get_contact_arrays_matrix(contacts, bin_size, chromos, chromo_limits):
 
  n, chromo_offsets, label_pos = _get_chromo_offsets(bin_size, chromos, chromo_limits)
  
  matrix = np.zeros((n, n), float)
     
  n_ambig = 0
  n_homolog = 0
  n_trans = 0
  n_cis = 0
  n_cont = 0
  
  for i, chr_1 in enumerate(chromos):
    for chr_2 in chromos[i:]:

      if chr_1 > chr_2:
        chr_a, chr_b = chr_2, chr_1
      else:
        chr_a, chr_b = chr_1, chr_2

      contact_matrix = contacts.get((chr_a, chr_b))

      if contact_matrix is None:
        continue
        
      count = contact_matrix.sum()
      
      if count == 0.0:
        continue
      
      bp_a, bin_a, size_a = chromo_offsets[chr_a]
      bp_b, bin_b, size_b = chromo_offsets[chr_b]

      sub_mat = _downsample_matrix(contact_matrix, (size_a, size_b))
      
      matrix[bin_a:bin_a+size_a,bin_b:bin_b+size_b] += sub_mat
      matrix[bin_b:bin_b+size_b,bin_a:bin_a+size_a] += sub_mat.T
      
      count = int(count)
      
      if chr_a != chr_b:
        if ('.' in chr_a) and ('.' in chr_b) and (chr_a.split('.')[0] == chr_b.split('.')[0]):
          n_homolog += count

        n_trans += count

      else:
        n_cis += count
      
      n_cont += count 
  
  return (n_cont, n_cis, n_trans, n_homolog, n_ambig), matrix, label_pos, chromo_offsets


def _limits_to_shape(limits_a, limits_b, bin_size):
  
  start_a, end_a = limits_a
  start_b, end_b = limits_b
  
  n = int(ceil(end_a/float(bin_size))) - int(start_a/bin_size)
  m = int(ceil(end_b/float(bin_size))) - int(start_b/bin_size)
  
  return n, m


def get_single_array_matrix(contact_matrix, limits_a, limits_b, is_cis, orig_bin_size, bin_size):
  
  if bin_size == orig_bin_size:
    if is_cis:
      a, b = contact_matrix.shape
      n = max(a, b)
      matrix = np.zeros((n,n), float)
      matrix[:a,:b] += contact_matrix
      matrix[:b,:a] += contact_matrix.T
 
    else:
      matrix = contact_matrix.astype(float)
  
  else:
    n, m = _limits_to_shape(limits_a, limits_b, bin_size)
    matrix = _downsample_matrix(contact_matrix, (n, m)).astype(float)
 
    if is_cis:
      matrix += matrix.T
  
  return matrix
  

def get_single_list_matrix(contact_list, limits_a, limits_b, is_cis, bin_size, ambig_groups):
  
  n, m = _limits_to_shape(limits_a, limits_b, bin_size)
  
  matrix = np.zeros((n, m), float)
  ambig_matrix = np.zeros((n, m), float)

  start_a, end_a = limits_a
  start_b, end_b = limits_b
  
  for p_a, p_b, nobs, ag in contact_list:
    a = int((p_a-start_a)/bin_size)
    b = int((p_b-start_b)/bin_size) 
    
    if ambig_groups[ag] > 1:
      ambig_matrix[a,b] += nobs
    else:
      matrix[a,b] += nobs
  
  if is_cis:
    matrix += matrix.T
    ambig_matrix += ambig_matrix.T
 
  return matrix, ambig_matrix
  
  
def get_contact_lists_matrix(contacts, bin_size, chromos, chromo_limits):
  
  n, chromo_offsets, label_pos = _get_chromo_offsets(bin_size, chromos, chromo_limits)
  
  # Fill contact map matrix, last dim is for (un)ambigous
  matrix = np.zeros((n, n), float)
  ambig_matrix = np.zeros((n, n), float)
  
  ambig_groups = defaultdict(int)
    
  for key in contacts:
    for p_a, p_b, nobs, ag in contacts[key]:
      ambig_groups[ag] += 1

  homolog_groups = set()
  trans_groups = set()
  cis_groups = set()
  
  for i, chr_1 in enumerate(chromos):
    for chr_2 in chromos[i:]:

      if chr_1 > chr_2:
        chr_a, chr_b = chr_2, chr_1
      else:
        chr_a, chr_b = chr_1, chr_2

      contact_list = contacts.get((chr_a, chr_b))

      if contact_list is None: # Nothing for this pair: common for single-cell Hi-C
        continue

      s_a, off_a, size_a = chromo_offsets[chr_a]
      s_b, off_b, size_b = chromo_offsets[chr_b]

      for p_a, p_b, nobs, ag in contact_list:
        if chr_a != chr_b:
          if ('.' in chr_a) and ('.' in chr_b) and (chr_a.split('.')[0] == chr_b.split('.')[0]):
            homolog_groups.add(ag)

          else:
            trans_groups.add(ag)

        else:
          cis_groups.add(ag)

        a = off_a + int((p_a-s_a)/bin_size)
        b = off_b + int((p_b-s_b)/bin_size)
 
        if ambig_groups[ag] == 1:
          matrix[a, b] += nobs
          matrix[b, a] += nobs
 
        else:
          ambig_matrix[a, b] += nobs
          ambig_matrix[b, a] += nobs
        
  n_ambig = len([x for x in ambig_groups.values() if x > 1])
  n_homolog = len(homolog_groups)
  n_trans = len(trans_groups)
  n_cis = len(cis_groups)
  n_cont = len(ambig_groups)
  
  return (n_cont, n_cis, n_trans, n_homolog, n_ambig), matrix, ambig_matrix, label_pos, chromo_offsets, ambig_groups


def _get_tick_delta(n, bin_size, max_ticks=10, unit=1e6):
  
  val_max = n * bin_size
  
  d = int(val_max/max_ticks)
  
  while d % 5 != 0:
    d += 1
 
  tick_delta = d/bin_size
  
  nminor = d/5

  return tick_delta, nminor
  

def plot_contact_matrix(matrix, bin_size, title, scale_label, chromo_labels=None, axis_chromos=None, grid=None,
                        stats_text=None, colors=None, bad_color='#404040', log=True, pdf=None,
                        watermark='nuc_tools.contact_map', legend=None, tracks=None, v_max=None, v_min=None,
                        ambig_matrix=None):
  
  from nuc_tools import util
  mmax = matrix.max()
  if not mmax:
    util.info('Map empty for ' + title, line_return=True)
    return
  
  if mmax < 0:
    matrix = -1 * matrix
  
  if not colors:
    if log or (matrix.min() < 0):
      colors = ['#0000B0', '#0080FF', '#FFFFFF', '#FF0000', '#800000']
    else:
      colors = ['#FFFFFF', '#0080FF' ,'#FF0000','#000000']  
  
  cmap = LinearSegmentedColormap.from_list(name='pcm', colors=colors, N=255)    
  cmap.set_bad(color=bad_color)

  if (ambig_matrix is not None) and ambig_matrix.max():
    do_ambig = True
    ambig_colors = ['#FFFFFF', '#D0D000', '#C0C000', '#B0B000']
    cmap2 = LinearSegmentedColormap.from_list(name='pcm', colors=ambig_colors, N=255)
    cmap2.set_bad(color=bad_color)
    
    ListedColormap
    clist = cmap(np.arange(cmap.N))
    clist[0,-1] = 0.0
    cmap = ListedColormap(clist)
    
  else:
    do_ambig = False  
   
  a, b = matrix.shape
  unit = 1e6 # Megabase
    
  if chromo_labels:
    xlabel_pos, xlabels = zip(*chromo_labels)
    ylabel_pos = xlabel_pos 
    ylabels = xlabels
    xrotation = 90.0
    xminor_tick_locator = None
    yminor_tick_locator = None
    
  else:
    xrotation = None
    tick_delta, nminor = _get_tick_delta(b, bin_size/unit)
    xlabel_pos = np.arange(0, b, tick_delta) # Pixel bins
    xlabels = ['%.1f' % (x*bin_size/unit) for x in xlabel_pos]
    xminor_tick_locator = AutoMinorLocator(nminor)
    
    tick_delta, nminor = _get_tick_delta(a, bin_size/unit) 
    ylabel_pos = np.arange(0, a, tick_delta) # Pixel bins
    ylabels = ['%.1f' % (y*bin_size/unit) for y in ylabel_pos]
    yminor_tick_locator = AutoMinorLocator(nminor)
   
  if 0: # tracks:
    n_tracks = len(tracks)
    h = [10] * n_tracks
    h.append(a)
    w = [1] * (n_tracks + 1)
    fig, axarr = plt.subplots(n_tracks+1, 1, gridspec_kw = {'height_ratios':h, 'width_ratios':w})
    ax = axarr[-1]
    colors = ['#FF0000','#FFFFFF']
    cmap_t = LinearSegmentedColormap.from_list(name='pcmt', colors=colors, N=255)
    cmap_t.set_under(color='#BBBBBB')
    
    for i, track in enumerate(tracks):
      track = np.array(track).reshape((1, a))
      axarr[i].matshow(track, interpolation='none', cmap=cmap_t, vmin=0.9, aspect=a/2)
      #axarr[i].plot(track)
  
  else:
    n_tracks = 0  
    fig, ax = plt.subplots(n_tracks+1, 1)
  
  if grid and grid is not True:
    grid = np.array(grid, float)
    ax.hlines(grid-0.5, -0.5, float(b), color='#B0B0B0', alpha=0.5, linewidth=0.1)
    ax.vlines(grid, float(a), -0.5, color='#B0B0B0', alpha=0.5, linewidth=0.1)
  
  if log:
    if do_ambig:
      cax2 = ax.matshow(ambig_matrix, interpolation='none', cmap=cmap2, norm=LogNorm(vmin=1), origin='upper')
                        
    cax = ax.matshow(matrix, interpolation='none', cmap=cmap, norm=LogNorm(vmin=1), origin='upper')
  
  else:
    if v_max is None:
      v_max = max(-matrix.min(), matrix.max())
    
    if v_min is None:
      v_min = -v_max
    
    if do_ambig:
      cax2 = ax.matshow(ambig_matrix, interpolation='none', cmap=cmap2, 
                        vmin=v_min, vmax=v_max, origin='upper')
    
    cax = ax.matshow(matrix, interpolation='none', cmap=cmap,
                     vmin=v_min, vmax=v_max, origin='upper') 
    
  ax.xaxis.tick_bottom()
  
  
  if chromo_labels and len(xlabels) > 25:
    ax.set_xticklabels(xlabels, fontsize=5, rotation=xrotation)
    ax.set_yticklabels(ylabels, fontsize=5)
  
  else:
    ax.set_xticklabels(xlabels, fontsize=9, rotation=xrotation)
    ax.set_yticklabels(ylabels, fontsize=9)
 
                
  ax.xaxis.set_ticks(xlabel_pos)
  ax.yaxis.set_ticks(ylabel_pos)
  
  ax.xaxis.set_tick_params(which='both', direction='out')
  ax.yaxis.set_tick_params(which='both', direction='out')
  
  if stats_text:
    ax.text(0, -int(1 + 0.01 * a), stats_text, fontsize=9)  
    
  ax.text(0.01, 0.01, watermark, color='#B0B0B0', fontsize=8, transform=fig.transFigure) 
  ax.set_title(title)
    
  if chromo_labels:
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('Chromosome')
    
  elif axis_chromos:
    ax.set_ylabel('Position %s (Mb)' % axis_chromos[0])
    ax.set_xlabel('Position %s (Mb)' % axis_chromos[1])
    ax.xaxis.set_minor_locator(xminor_tick_locator)
    ax.yaxis.set_minor_locator(yminor_tick_locator)
    
    if grid is True and not log:
      ax.grid(alpha=0.08, linestyle='-', linewidth=0.1)
  
  else:
    ax.set_xlabel('Position (Mb)')
    ax.set_ylabel('Position (Mb)')
    ax.xaxis.set_minor_locator(xminor_tick_locator)
    ax.yaxis.set_minor_locator(yminor_tick_locator)
    
    if grid is True and not log:
      ax.grid(alpha=0.08, linestyle='-', linewidth=0.1)
 
  if legend:
    for label, color in legend:
      ax.plot([], linewidth=3, label=label, color=color)
    
    ax.legend(fontsize=8, loc=9, ncol=len(legend), bbox_to_anchor=(0.5, 1.05), frameon=False)
  
  
  if do_ambig:
    cbar2 = plt.colorbar(cax2, shrink=0.3,  use_gridspec=False, anchor=(-1.225, 0.2))
    cbar2.ax.tick_params(labelsize=8)
    cbar2.set_label('Ambig. count', fontsize=11)
    
  cbar = plt.colorbar(cax, shrink=0.3, use_gridspec=False, anchor=(0.0, 0.8))
  cbar.ax.tick_params(labelsize=8)
  cbar.set_label(scale_label, fontsize=11)
    
  dpi= 4 * int(float(a)/(fig.get_size_inches()[1]*ax.get_position().size[1]))
  util.info(' .. making map ' + title + ' (dpi=%d)' % dpi, line_return=True)

  if pdf:
    pdf.savefig(dpi=dpi)
  else:
    plt.show()
    
  plt.close()
  
                
def contact_map(in_path, out_path, bin_size=None, bin_size2=250.0, bin_size3=500.0,
                no_separate_cis=False, separate_trans=False, show_chromos=None,
                use_corr=False, is_single_cell=False, screen_gfx=False, black_bg=False,
                font=None, font_size=12, line_width=0.2, min_contig_size=None, chromo_grid=False):
  
  from nuc_tools import io, util
  from formats import ncc, npz
  
  if out_path:
    file_root, file_ext = os.path.splitext(out_path)
    file_ext = file_ext.lower()
    
    if file_ext == '.pdf':
      out_path = file_root + '.pdf'
    
  else:
    file_root, file_ext = os.path.splitext(in_path)
    
    if file_ext.lower() == '.gz':
      file_root, file_ext = os.path.splitext(file_root)
    
    out_path = file_root + '.pdf'
  
  if screen_gfx:
    util.info('Displaying contact map for {}'.format(in_path))
  else:
    util.info('Making PDF contact map for {}'.format(in_path))
  
  if in_path.lower().endswith('.ncc') or in_path.lower().endswith('.ncc.gz'):
    file_bin_size = None
    chromosomes, chromo_limits, contacts = ncc.load_file(in_path)
    
  else:
    file_bin_size, chromo_limits, contacts = npz.load_npz_contacts(in_path)

  if not chromo_limits:
    util.critical('No chromosome contact data read')

  if min_contig_size:
    min_contig_size = int(min_contig_size * 1e6)
  else:
    largest = max([e-s for s, e in chromo_limits.values()])
    min_contig_size = int(DEFAULT_SMALLEST_CONTIG*largest) 
    util.info('Min. contig size not specified, using {}% of largest: {:,} bp'.format(DEFAULT_SMALLEST_CONTIG*100, min_contig_size))
  
  if show_chromos:
    chr_names = ', '.join(sorted(chromo_limits))
    
    filtered = {}
    found = set()
    for chromo, lims in chromo_limits.items():
      if chromo in show_chromos:
        filtered[chromo] = lims
        found.add(chromo)
        
      elif chromo.lower().startswith('chr') and (chromo[3:] in show_chromos):
        filtered[chromo] = lims
        found.add(chromo[3:])
    
    unknown = sorted(set(show_chromos) - found)
         
    chromo_limits = filtered
  
    if not chromo_limits:
      util.critical('Chromosome selection doesn\'t match any in the contact file. Available: {}'.format(chr_names))
    elif unknown:
      util.warn('Some selected chromosomes don\'t match the contact file: {}'.format(', '.join(unknown)))
    
  if not bin_size:
    bin_size = DEFAULT_SC_MAIN_BIN_KB if is_single_cell else DEFAULT_MAIN_BIN_KB
  
  if not bin_size2:
    bin_size2 = DEFAULT_SC_CHR_BIN_KB if is_single_cell else DEFAULT_CIS_BIN_KB

  if not bin_size3:
    bin_size3 = DEFAULT_SC_CHR_BIN_KB if is_single_cell else DEFAULT_TRANS_BIN_KB
      
  """
  tot_size = 0
  
  for chromo in chromo_limits:
    s, e = chromo_limits[chromo]
    size = e-s
    
    if size >= min_contig_size:
      tot_size += size 
  
  bin_size = int(tot_size/1000)
  util.info('Bin size not specified, using approx. 1000 x 1000 bin equivalent: {:,} bp'.format(bin_size))
  """
    
  separate_cis = not bool(no_separate_cis)
  bin_size = int(bin_size * 1e3)
  bin_size2 = int(bin_size2 * 1e3)
  bin_size3 = int(bin_size3 * 1e3)
          
  # Get sorted chromosomes, ignore small contigs as appropriate
  chromos = []
  skipped = []
  for chromo in chromo_limits:
    s, e = chromo_limits[chromo]

    if (e-s) < min_contig_size:
      if show_chromos and (chromo in show_chromos):
        msg = 'Chromosome {} is below the size limit but was nonethless included as it was included in the -chr option'
        util.info(msg.format(chromo))        
      else:
        skipped.append(chromo)
        continue

    chromos.append(chromo)

  if skipped:
    util.info('Skipped {:,} small chromosomes/contigs < {:,} bp'.format(len(skipped), min_contig_size))

  chromos = util.sort_chromosomes(chromos)
   
  chromo_labels = []
  for chromo in chromos:
    if chromo.upper().startswith('CHR'):
      chromo = chromo[3:]
    chromo_labels.append(chromo)

  if file_bin_size:
    count_list, full_matrix, label_pos, offsets = get_contact_arrays_matrix(contacts, bin_size, chromos, chromo_limits)
    n_cont, n_cis, n_trans, n_homolog, n_ambig = count_list
    ambig_matrix = None
    
  else:
    count_list, full_matrix, ambig_matrix, label_pos, offsets, ambig_groups = get_contact_lists_matrix(contacts, bin_size, chromos, chromo_limits)
    n_cont, n_cis, n_trans, n_homolog, n_ambig = count_list
  
  if use_corr:
    has_neg = True
    full_matrix = get_corr_mat(full_matrix)
  else:
    has_neg = full_matrix.min() < 0
  
  max_val = full_matrix.max()
  
  if full_matrix.sum() < 1e7 and not is_single_cell:
    util.warn('Contact map is sparse but single-cell "-sc" option not used')

  if has_neg:
    use_log = False
  elif is_single_cell:
    use_log = False
  else:
    use_log = True
  
  n = len(full_matrix)
  util.info('Full contact map size %d x %d' % (n, n))
  
  f_cis = 100.0 * n_cis / float(n_cont or 1)
  f_trans = 100.0 * n_trans / float(n_cont or 1)

  if use_corr:
    metric = 'Correlation'
    v_max = 0.5
    v_min = -0.5
  else:
    metric = 'Count'
    
    if is_single_cell:
      v_max = 4
      v_min = 0
    else:
      v_max = None
      v_min = 0.0
  
  if has_neg and not use_corr:
    stats_text = ''
    metric = 'Value '
  
  elif n_homolog:
    f_homolog = 100.0 * n_homolog / float(n_cont or 1)  
    stats_text = 'Contacts:{:,d} cis:{:,d} ({:.1f}%) trans:{:,d} ({:.1f}%) homolog:{:,d} ({:.1f}%)'
    stats_text = stats_text.format(n_cont, n_cis, f_cis, n_trans, f_trans, n_homolog, f_homolog)
  
  else:
    stats_text = 'Contacts:{:,d} cis:{:,d} ({:.1f}%) trans:{:,d} ({:.1f}%)'
    stats_text = stats_text.format(n_cont, n_cis, f_cis, n_trans, f_trans)

  if black_bg:
    if has_neg:
      colors = ['00FFFF', '#0000FF', '#000000', '#FF0000', '#FFFF00']
    else:
      colors = ['#000000', '#BB0000', '#DD8000', '#FFFF00', '#FFFF80','#FFFFFF']
    
    bad_color = '#404040'

  else:
    if has_neg:
      colors = ['#0000B0', '#0080FF', '#FFFFFF', '#FF0000', '#800000']
    elif is_single_cell:
      colors = ['#FFFFFF', '#0080FF' ,'#0000FF','#0000B0','#000080']
    
    else:
      colors = ['#FFFFFF', '#0080FF' ,'#FF0000','#000000']
    
    bad_color = '#B0B0B0'

  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(out_path)
  
  title = os.path.basename(in_path)
  grid = [offsets[c][1] for c in chromos[1:]]
  scale_label = '%s (%.2f Mb bins)' % (metric, bin_size/1e6)
  
  plot_contact_matrix(full_matrix, bin_size, title, scale_label, zip(label_pos, chromo_labels),
                      None, grid, stats_text, colors, bad_color, log=use_log, pdf=pdf,
                      v_max=v_max, v_min=v_min, ambig_matrix=ambig_matrix) 
  use_log
  if separate_cis or separate_trans:
  
    pairs = []
    if separate_cis:
      for chr_a in chromos:
        pair = (chr_a, chr_a)
        
        if pair in contacts:
          pairs.append(pair)
    
    if separate_trans:
      for i, chr_a in enumerate(chromos[:-1]):
        for chr_b in chromos[i+1:]:
          pair = (chr_a, chr_b)
          
          if pair in contacts:
            pairs.append(pair)
    
    for pair in pairs:
      chr_a, chr_b = pair # Display order
      is_cis = chr_a == chr_b
      key = tuple(sorted(pair)) # Key order
      limits_a = chromo_limits[key[0]]
      limits_b = chromo_limits[key[1]]
      
      pair_bin_size = bin_size2 if is_cis else bin_size3
              
      if file_bin_size:
        matrix = get_single_array_matrix(contacts[key], limits_a, limits_b, is_cis, file_bin_size, pair_bin_size)
        ambig_matrix = None
      else:
        matrix, ambig_matrix = get_single_list_matrix(contacts[key], limits_a, limits_b, is_cis, pair_bin_size, ambig_groups)
      
      if key != pair:
        matrix = matrix.T
            
      if use_corr:
        if is_cis:
          matrix = get_corr_mat(matrix)
        else:
          if file_bin_size:
            matrix_a = get_single_array_matrix(contacts[(chr_a, chr_a)], limits_a, limits_a, True, file_bin_size, pair_bin_size)
          else:
            matrix_a, amb_a = get_single_list_matrix(contacts[(chr_a, chr_a)], limits_a, limits_a, True, pair_bin_size, ambig_groups)

          if file_bin_size:
            matrix_b = get_single_array_matrix(contacts[(chr_b, chr_b)], limits_b, limits_b, True, file_bin_size, pair_bin_size)
          else:
            matrix_b, amb_b = get_single_list_matrix(contacts[(chr_b, chr_b)], limits_b, limits_b, True, pair_bin_siz, ambig_groupse)

          matrix = get_trans_corr_mat(matrix_a, matrix_b, matrix)
             
      title = 'Chromosome %s' % chr_a if is_cis else 'Chromosomes %s - %s ' % pair
      
      if is_cis:
        scale_label = '%s (%.1f kb bins)' % (metric, pair_bin_size/1e3)
      else:
        scale_label = '%s (%.3f Mb bins)' % (metric, pair_bin_size/1e6)
     
      plot_contact_matrix(matrix, pair_bin_size, title, scale_label, None, pair,
                          chromo_grid, None, colors, bad_color, log=use_log, pdf=pdf,
                          v_max=v_max, v_min=v_min, ambig_matrix=ambig_matrix)
                        
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

  arg_parse.add_argument(metavar='CONTACT_FILE', nargs='+', dest='i',
                         help='Input NPZ (binned, bulk Hi-C data) or NCC format (single-cell) chromatin contact file(s). Wildcards accepted.')

  arg_parse.add_argument('-o', metavar='OUT_FILE', nargs='+', default=None,
                         help='Optional output file name. If not specified, a default based on the input file name and output format will be used. ' \
                              'If multiple input contact files are specified there must be one output for each input.')
  
  arg_parse.add_argument('-chr', metavar='CHROMOSOMES', nargs='+', default=None,
                         help='Optional selection of chromsome names to generate contact maps for.')

  arg_parse.add_argument('-sc', '--single-cell', default=False, action='store_true', dest="sc",
                         help='Specifies that the input data is from single-cell Hi-C')

  arg_parse.add_argument('-nc', '--no-cis', default=False, action='store_true', dest="nc",
                         help='Do not display separate contact maps for individual chromosomes (intra-chromosomal contacts). ' \
                              'Only the overall whole-genome map will be displayed (unless -t option also used).')

  arg_parse.add_argument('-t', '--trans', default=False, action='store_true', dest="t",
                         help='Display separate contact maps for all trans (inter-chromosomal) pairs. ' \
                              'By default the overall whole-genome and intra-chromosome maps are generated.')

  arg_parse.add_argument('-g', default=False, action='store_true',
                         help='Display graphics on-screen using matplotlib, where possible and do not automatically save output.')

  arg_parse.add_argument('-s1', '--bin-size-main', default=None, metavar='BIN_SIZE', type=float, dest="s1",
                         help='Binned sequence region size (the resolution) for the overall, whole-genome contact map, in kilobases. ' \
                              'Default is {:.1f} kb or {:.1f} kb if single-cell "-sc" option used.'.format(DEFAULT_MAIN_BIN_KB, DEFAULT_SC_MAIN_BIN_KB))

  arg_parse.add_argument('-s2', '--bin-size-cis', default=None, metavar='BIN_SIZE', type=float, dest="s2",
                         help='Binned sequence region size (the resolution) for separate intra-chromsomal maps, ' \
                              'in kilobases. Default is {:.1f} kb or {:.1f} kb if single-cell "-sc" option used..'.format(DEFAULT_CIS_BIN_KB, DEFAULT_SC_CHR_BIN_KB))
  
  arg_parse.add_argument('-s3', '--bin-size-trans', default=None, metavar='BIN_SIZE', type=float, dest="s3",
                         help='Binned sequence region size (the resolution) for separate inter-chromsomal maps, ' \
                              'in kilobases. Default is {:.1f} kb. or {:.1f} kb if single-cell "-sc" option used.'.format(DEFAULT_TRANS_BIN_KB, DEFAULT_SC_CHR_BIN_KB))

  arg_parse.add_argument('-m', default=0.0, metavar='MIN_CONTIG_SIZE', type=float,
                         help='The minimum chromosome/contig sequence length in Megabases for inclusion. ' \
                              'Default is {}%% of the largest chromosome/contig length.'.format(DEFAULT_SMALLEST_CONTIG*100))

  arg_parse.add_argument('-b', '--black-bg',default=False, action='store_true', dest="b",
                         help='Specifies that the contact map should have a black background (default is white).')

  arg_parse.add_argument('-corr', default=False, action='store_true', dest="corr",
                         help='Plot Pearson correlation coefficients for the contacts, rather than counts. ' \
                              'For trans/inter-chromosome pairs, the correlations shown are the non-cis part of the '\
                              'square, symmetric correlation matrix of the combined map for both chromosomes.')

  arg_parse.add_argument('-grid', default=False, action='store_true', dest="grid",
                         help='Show grid lines at numeric chromosome positions.')
                         
  args = vars(arg_parse.parse_args(argv))

  in_paths = args['i']
  out_paths = args['o']
  screen_gfx = args['g']
  bin_size = args['s1']
  bin_size2 = args['s2']
  bin_size3 = args['s3']
  min_contig_size = args['m']
  black_bg = args['b']
  no_sep_cis = args['nc']
  sep_trans = args['t']
  chromos = args['chr']
  use_corr = args['corr']
  is_single = args['sc']
  chromo_grid = args['grid']
  
  if not in_paths:
    arg_parse.print_help()
    sys.exit(1)
  
  if out_paths:
    if len(out_paths) != len(in_paths):
      util.critical('The number of output file paths does not match the number input')
      
    if screen_gfx:
      util.warn('Output files will not be written in screen graphics (-g) mode')
      out_paths = [None] * len(in_paths)
      
  else:
    out_paths = [None] * len(in_paths)

  for in_path, out_path in zip(in_paths, out_paths):
    if not os.path.exists(in_path):
      util.critical('Input contact file could not be found at "{}"'.format(in_path))

    contact_map(in_path, out_path, bin_size, bin_size2, bin_size3,
                no_sep_cis, sep_trans, chromos, use_corr, is_single,
                screen_gfx, black_bg, min_contig_size=min_contig_size,
                chromo_grid=chromo_grid)


if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()

"""
For single-cell add Mitosis and ploidy scores
Tweak title and info positions
"""


  
