import os, sys, math
import numpy as np
from glob import glob
from os.path import dirname

np.seterr(all='raise')
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

PROG_NAME = 'data_track_compare'
VERSION = '1.0.0'
DESCRIPTION = 'Plot and measure similarities between data tracks in BED format'

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import LinearSegmentedColormap, Normalize

DEFAULT_PDF_OUT = 'dtc_out_job{}_D{}x{}.pdf'
DEFAULT_BIN_KB  = 20.0
DEFAULT_SEQ_REG_KB = 10
PDF_DPI = 200
  
HIST_BINS2D = 50

LOG_THRESH = 4
INF = float('inf')

NUM_QUANT_BINS = 10

PLOT_CMAP = LinearSegmentedColormap.from_list(name='PLOT_CMAP', colors=['#FF0000','#0050FF','#BBBB00','#808080','#FF00FF', '#00BBBB', '#00BB00', '#8000FF', '#FF8000'], N=255)   

def _load_bin_data(data_paths, bin_size):
  
  from nuc_tools import util
  from formats import bed
  
  data_dict = {}
  binned_data_dict = {}
  
  for data_bed_path in set(data_paths):
    util.info(' .. load {}'.format(data_bed_path), line_return=True)
    data_regions, data_values = bed.load_bed_data_track(data_bed_path)[:2]
    data_dict[data_bed_path] = data_regions, data_values 
    
  util.info('Loaded {} data tracks'.format(len(data_dict)))
  
  chromo_limits = {}
  
  for data_bed_path in data_dict:
    data_regions, data_values = data_dict[data_bed_path]
    
    for chromo in data_regions:
      regions = data_regions[chromo]
      s = regions.min()
      e = regions.max()
      
      if chromo in chromo_limits:
        a, b = chromo_limits[chromo]
        chromo_limits[chromo] = (min(a, s), max(b, e))
      else:
        chromo_limits[chromo] = (s, e)

  util.info('Found {} chromosomes/contigs'.format(len(chromo_limits)))
  
  chromos = sorted(chromo_limits)
      
  for data_bed_path in data_dict:
    data_regions, data_values = data_dict[data_bed_path]
    hists = []
    
    for chromo in chromos:
      start, end = chromo_limits[chromo]
      hist = util.bin_region_values(data_regions.get(chromo, []),
                                    data_values.get(chromo, []),
                                    bin_size, start, end)
      hists.append(hist)
                                                  
    binned_data_dict[data_bed_path] = np.concatenate(hists, axis=0)
    util.info(' .. bin {}'.format(data_bed_path), line_return=True)
  
  util.info('Binned data tracks into {:,} regions'.format(len(binned_data_dict[data_bed_path])))
                                                   
  return binned_data_dict, data_dict, chromo_limits


def plot_value_distribs(data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, pdf):
  
  data_labels = []
  data_paths = []

  for i, ref_data_path in enumerate(ref_data_paths):
    if ref_data_path not in data_paths:
      data_paths.append(ref_data_path)
      data_labels.append(ref_labels[i])

  for i, comp_data_path in enumerate(comp_data_paths):
    if ref_data_path not in data_paths:
      data_paths.append(comp_data_path)
      data_labels.append(comp_labels[i])
  
  n_data = len(data_paths)
  n_rows = int(math.ceil(math.sqrt(n_data)))
  n_cols = int(math.ceil(n_data / float(n_rows))) 
  
  fig, axarr = plt.subplots(n_rows, n_cols, squeeze=False) # , sharex=True, sharey=True)
    
  plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.94, wspace=0.3, hspace=0.2)
  
  for i, d1 in enumerate(data_paths):
    row = int(i//n_cols)
    col = i % n_cols
    
    data_regions, data_values = data_dict[d1]
    
    vals = np.concatenate([data_values[chromo] for chromo in data_values])
    vals = vals[vals > 0] 
    vals = np.log10(vals)
    label = data_labels[i]
    
    ax = axarr[row, col]
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)    
    
    hist, edges = np.histogram(vals, bins=100)
    
    ax.plot(edges[:-1], hist, alpha=0.67, color=PLOT_CMAP(i/float(n_data-1)), label=label + '\nn=%d' % len(vals))
    ax.legend(fontsize=7)

  fig.text(0.5, 0.975, 'Data value distributions', color='#000000',
           horizontalalignment='center',
           verticalalignment='center', rotation=0,
           fontsize=13, transform=fig.transFigure)
    
  fig.text(0.025, 0.5, 'Count', color='#000000',
           horizontalalignment='center',
           verticalalignment='center', rotation=90,
           fontsize=13, transform=fig.transFigure)

  fig.text(0.5, 0.025, '$log_{10}$(Data value)', color='#000000',
           horizontalalignment='center',
           verticalalignment='center', rotation=0,
           fontsize=13, transform=fig.transFigure)
            
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show()
    
  plt.close()     
    
  
def plot_data_line_correlations(data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, pdf):

  from scipy.stats import rankdata, sem
  
  n_ref = len(ref_data_paths)
  n_comp = len(comp_data_paths)
  
  n_rows = int(math.ceil(math.sqrt(n_ref)))
  n_cols = int(math.ceil(n_ref / float(n_rows)))
  
  fig, axarr = plt.subplots(n_rows, n_cols, squeeze=False) # , sharex=True, sharey=True)
    
  plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)

  fig.text(0.5, 0.95, 'Data quantile means ({:,} kb bins)'.format(bin_size/1e3), color='#000000',
          horizontalalignment='center',
          verticalalignment='center',
          fontsize=13, transform=fig.transFigure)
  
  fig.set_size_inches(3*n_cols, 2*n_rows)
  
  nq = NUM_QUANT_BINS
  x_label = 'Precentile bin'
  y_max_all = 0.0
  y_min_all = 0.0
  used_labels = set()
  used = np.zeros((n_rows, n_cols))
  
  for i, d1 in enumerate(ref_data_paths):
    row = int(i//n_cols)
    col = i % n_cols
     
    used[row, col] = 1
   
    vals1 = data_dict[d1]
    nzy = vals1 > 0

    mn, md, mx  = np.percentile(vals1[nzy], [0.5, 50.0, 99.5])    
    #y_lim = (np.log10(mn), np.log10(mx))
 
    ax = axarr[row, col]
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    #ax.set_ylim(*y_lim)
    
    x_lim = (0.0, 100.0)
    y_max = 0.0
    y_min = INF
    
    if i == 0:
      for j, d2 in enumerate(comp_data_paths):
        color = PLOT_CMAP(j/float(n_comp-1))
        label = comp_labels[j]
        ax.plot([], alpha=0.65, color=color, label=label)    
    
    for j, d2 in enumerate(comp_data_paths):
      if j == i:
        continue
     
      color = PLOT_CMAP(j/float(n_comp-1))    
      
      vals2 = data_dict[d2]
      nzx = vals2 > 0
      nz = nzx & nzy
      
      y_vals = vals1[nz]
      x_vals = vals2[nz]
      
      y_vals = np.log10(y_vals)
      
      x_vals = rankdata(x_vals)
      x_vals /= float(len(x_vals))
      idx = x_vals.argsort()
      x_vals = x_vals[idx]
      y_vals = y_vals[idx]
     
 
      bw = 100/nq
 
      y_split = np.array_split(y_vals, nq)
 
      x_pos = np.arange(0, nq)
      
      y_med = np.array([np.mean(d) for d in y_split])
      
      y_sem = np.array([sem(d) for d in y_split])
     
      y_upp = y_med + y_sem
      y_low = y_med - y_sem
     
      y_max = max(y_max, y_upp.max())
      y_min = min(y_min, y_low.max())
       
      #y_q25 = [np.percentile(d, 25.0) for d in y_split]
      #y_q75 = [np.percentile(d, 75.0) for d in y_split]
      
      #ax.plot(x_pos, y_med, label=comp_labels[col], alpha=0.5)

      ax.errorbar(x_pos, y_med, y_sem, linewidth=1, alpha=0.4, color=color, zorder=j+n_comp)
      
      #ax.fill_between(x_pos, y_q25, y_q75, linewidth=0, alpha=0.2, color=color, zorder=j)
      
    y_max_all = max(y_max_all, y_max)
    y_min_all = max(y_min_all, y_max)
    ax.set_ylabel(ref_labels[i], fontsize=9)
    
    if row == n_rows-1:
      ax.set_xticks(np.arange(0, nq+1)-0.5)
      ax.set_xticklabels(bw*np.arange(0, nq+1))
      ax.set_xlabel(x_label, fontsize=9)
    else:
      ax.set_xticks([])
   
  fig.legend(frameon=False, fontsize=9, loc=8, ncol=int((n_ref+1)/2))

  for col in range(n_cols):
    lr = used[:,col].sum()-1
    
    for row in range(n_rows):
      ax = axarr[row, col]
      ax.grid(True, which='major', axis='x', alpha=0.5, linestyle='--')
      #ax.set_ylim(y_min_all, 1.1*y_max_all)
      
      if row < lr:
        ax.set_xticklabels([])
         
      elif row == lr:
        ax.set_xlabel(x_label, fontsize=8)        
      
      else:
        ax.axis('off')
  
  fig.text(0.025, 0.5, 'Mean $log_{10}$(Data value)', color='#000000',
           horizontalalignment='center',
           verticalalignment='center', rotation=90,
           fontsize=13, transform=fig.transFigure)
 
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show()
    
  plt.close()  
  

def plot_data_correlations(title, data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, colors, pdf,
                           quantile_x=False, quantile_y=False, violin=False, boxplot=False):
  
  from scipy.stats import sem, rankdata, pearsonr, spearmanr
  
  n_ref = len(ref_data_paths)
  n_comp = len(comp_data_paths)
  
  fig, axarr = plt.subplots(n_ref, n_comp, squeeze=False)
  
  fig.set_size_inches(max(8, 2*n_comp), max(8, 2*n_ref))
  
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.90, top=0.9, wspace=0.08, hspace=0.08)

  fig.text(0.5, 0.95, 'Data value {} plots ({:,} kb bins)'.format(title, bin_size/1e3), color='#000000',
          horizontalalignment='center',
          verticalalignment='center',
          fontsize=13, transform=fig.transFigure)
                
  if quantile_x:
    fig_xlabel = 'Data percentile'
    fig_ylabel = '$log_{10}$(Data value)'
  
  elif violin:
    fig_xlabel = 'Decile bin'
    fig_ylabel = '$log_{10}$(Data value)'
  
  elif boxplot:
    fig_xlabel = 'Decile bin'
    fig_ylabel = '$log_{10}$(Data value)'
  
  else:
    fig_xlabel = '$log_{10}$(Data value)'
    fig_ylabel = '$log_{10}$(Data value)'
  
  for row, d1 in enumerate(ref_data_paths):
    vals1 = data_dict[d1]
    nzy = vals1 > 0
    
    mn, md, mx  = np.percentile(vals1[nzy], [0.5, 50.0, 99.5])
    log_y = True # mx > LOG_THRESH * md
    
    if quantile_y:
      log_y = False
      y_lim = (0.0, 100.0)
      
    else:
      if log_y:
        y_lim = (np.log10(mn), np.log10(mx))
      else:
        y_lim = (mn, mx) 
    
    for col, d2 in enumerate(comp_data_paths):
      vals2 = data_dict[d2]
      nzx = vals2 > 0
      
      mn, md, mx  = np.percentile(vals2[nzx], [0.5, 50.0, 99.5])
      
      if quantile_x:
        x_lim = (0.0, 100.0)
        log_x = False
      else:
        log_x = True #  mx > LOG_THRESH * md
 
        if log_x:
          x_lim = (np.log10(mn), np.log10(mx))
        else:
          x_lim = (mn, mx)
 
      nz = nzx & nzy
      ax = axarr[row, col]
      ax.tick_params(axis='both', which='major', labelsize=7)
      ax.tick_params(axis='both', which='minor', labelsize=7)
      
      y_vals = vals1[nz]
      x_vals = vals2[nz]
      
      n = len(x_vals)  
      r, p = pearsonr(x_vals, y_vals)
      rho, p = spearmanr(x_vals, y_vals)
      
      if log_y:
        y_vals = np.log10(y_vals)
      elif quantile_y:
        y_vals = rankdata(y_vals)
        y_vals *= 100.0 / float(len(y_vals))
        
      if violin or boxplot:
        color = PLOT_CMAP(col/float(n_comp-1))
        x_vals = rankdata(x_vals)
        x_vals /= float(len(x_vals))
        idx = x_vals.argsort()
        x_vals = x_vals[idx]
        y_vals = y_vals[idx]
        
        bw = 100/NUM_QUANT_BINS
         
        y_split = np.array_split(y_vals, NUM_QUANT_BINS)
 
        if violin:
          x_pos = np.arange(0, NUM_QUANT_BINS)
          vp = ax.violinplot(y_split, x_pos, points=25, widths=0.8, bw_method=0.25,
                             showmeans=False, showextrema=False, showmedians=True)
          ax.set_xlim((-1, NUM_QUANT_BINS))
          ax.set_xticks(np.arange(0, NUM_QUANT_BINS+1)-0.5)
          ax.set_xticklabels(bw*np.arange(0, NUM_QUANT_BINS+1), rotation=90.0)
          
          for i, a in enumerate(vp['bodies']):
            a.set_facecolor(color)
            a.set_edgecolor('black')
 
          vp['cmedians'].set_color('black')
          
        else:
          bp = ax.boxplot(y_split, notch=True, sym=',', widths=0.8, whis=[10,90],
                          boxprops={'color':color},
                          capprops={'color':color},
                          whiskerprops={'color':color},
                          flierprops={'color':color, 'markeredgecolor':color, },
                          medianprops={'color':'#000000'})
        
          ax.set_xlim((0, NUM_QUANT_BINS+1))
          ax.set_xticks(0.5 + np.arange(0, NUM_QUANT_BINS+1))
          ax.set_xticklabels(bw*np.arange(0, NUM_QUANT_BINS+1), rotation=90.0)
        
        cax = None
                     
      else:

        if log_x:
          x_vals = np.log10(x_vals)
        elif quantile_x:
          x_vals = 100 * rankdata(x_vals)
          x_vals /= float(len(x_vals))
          
        ax.hist2d(x_vals, y_vals, 
                  bins=HIST_BINS2D, range=(x_lim, y_lim),
                  cmap=colors, vmin=0.0)
 
       
        ax.set_xlim(*x_lim)
        
      ax.set_ylim(*y_lim) 
       
      if row == n_ref-1:
        ax.set_xlabel(comp_labels[col], fontsize=9)
      else:
        ax.set_xticks([])
       
      if row == 0:
        ax.set_title(comp_labels[col], fontsize=9)
             
      if col == 0:
        ax.set_ylabel(ref_labels[row], fontsize=9)
      else:
        ax.set_yticks([])
        
      ax.text(0.05, 0.95, 'n={:,}\nr={:.2f}\n$\\rho$={:.2f}'.format(n,r,rho),
              color='#202020', verticalalignment='top',
              alpha=1.0, fontsize=8, transform=ax.transAxes)
  
  
  if not (violin or boxplot):
    cbax1 = fig.add_axes([0.91, 0.6, 0.02, 0.3]) # left, bottom, w, h
    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = ScalarMappable(norm=norm, cmap=colors)
    sm.set_array(np.linspace(0.0, 1.0, 10))
    
    cbar1 = plt.colorbar(sm, cax=cbax1, orientation='vertical')
  
    cbar1.ax.tick_params(labelsize=8)
    cbar1.set_label('Count/maximum', fontsize=9)
 

  fig.text(0.5, 0.025, fig_xlabel, color='#000000',
           horizontalalignment='center',
           verticalalignment='center',
           fontsize=13, transform=fig.transFigure)

  fig.text(0.025, 0.5, fig_ylabel, color='#000000',
           horizontalalignment='center',
           verticalalignment='center', rotation=90,
           fontsize=13, transform=fig.transFigure)
 
 
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show()
    
  plt.close()  


def _get_anchor_mat(ref_region_dict, ref_hist_dict, track_hist_dict, chromo_limits, bin_size, n_bins):
  
  
  hist_mat = []
  scores = []  
  
  hw = int(n_bins/2)
  
  didx = np.arange(n_bins)-hw
  
  w = np.linspace(-3, 3, n_bins)
  w = np.exp(-w*w)
  
  for chromo in ref_region_dict:
    if not len(ref_region_dict[chromo]):
      continue
      
    start, end = chromo_limits[chromo]
    middles = ref_region_dict[chromo].mean(axis=1).astype(int) # Region centers
    hist = track_hist_dict[chromo]
    m = len(hist)
    n = len(middles)
    
    idx =  ((middles-start)//bin_size).astype(int) # Indices of region centres in chromo track hist
    
    # For each need cut-out region of other track as hist - then stacked
    
    ranges = idx[:,None] + didx
    ranges = ranges.ravel()
    
    overhang = (ranges < 0) | (ranges > m-1)
    ranges[overhang] = 0
    
    mat = hist[ranges]
    mat[overhang] = 0
    mat = mat.reshape(n, n_bins)
    
    #nz = mat.max(axis=1 > 0)
    #mat = mat[nz]
    
    #mx = mat.argmax(axis=1)
    #sc = mv * (hw - np.abs(mx-hw))
    #sc = (mat * w).sum(axis=1)
    
    #ref_hist = ref_hist_dict[chromo]
    #ref_mat = ref_hist[ranges]
    #ref_mat[overhang] = 0
    #ref_mat = ref_mat.reshape(n, n_bins)
    
    sc = mat.max(axis=1)
    
    #mat /= mv[:,None]
        
    scores.append(sc)
    hist_mat.append(mat)
  
  hist_mat = np.concatenate(hist_mat, axis=0)
  scores = np.concatenate(scores, axis=0)
  
  hist_mat = hist_mat[scores.argsort()[::-1]]
      
  
  return hist_mat
  
  
def plot_seq_anchor_mat(raw_data_dict, chromo_limits, ref_data_paths, comp_data_paths, ref_labels, comp_labels,
                        colors, pdf, seq_reg_size, n_bins=100):
  
  from nuc_tools import util
  
  bin_size = (seq_reg_size*1000)/n_bins
  
  n_ref = len(ref_data_paths)
  n_comp = len(comp_data_paths)
  
  fig, axarr = plt.subplots(n_comp, n_ref, squeeze=False)
  
  fig.set_size_inches(max(8, 2*n_ref), max(8, 2*n_comp))
  
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.08, hspace=0.08)

  fig.text(0.5, 0.95, 'Anchored data density ({:,} bp bins)'.format(bin_size), color='#000000',
          horizontalalignment='center',
          verticalalignment='center',
          fontsize=13, transform=fig.transFigure)
  
  track_hist_dicts = {}
  
  track_hist_dicts = {}
  
  for d2 in comp_data_paths:
    region_dict, value_dict = raw_data_dict[d2]
    track_hist_dicts[d2] = {}
    meds = []
    
    for chromo in chromo_limits:
      start, end = chromo_limits[chromo]
      track_hist = np.log10(1.0+ util.bin_region_values(region_dict[chromo], value_dict[chromo], bin_size, start, end)) 
      track_hist_dicts[d2][chromo] = track_hist
      
      idx = (track_hist>0).nonzero()[0]
      
      if len(idx):
        meds.append(np.median(track_hist[idx]))
    
    med = np.median(meds)
    for chromo in chromo_limits:
      track_hist = track_hist_dicts[d2][chromo]
      track_hist_dicts[d2][chromo] = track_hist/med
       
  for col, d1 in enumerate(ref_data_paths):
    ref_region_dict, ref_value_dict = raw_data_dict[d1]
    
    ref_hist_dict = {}
    meds = []
    
    for chromo in chromo_limits:
      start, end = chromo_limits[chromo]
      ref_hist = np.log10(1.0+ util.bin_region_values(ref_region_dict[chromo], ref_value_dict[chromo], bin_size, start, end)) 
      ref_hist_dict[chromo] = ref_hist
      idx = (ref_hist>0).nonzero()[0]
      
      if len(idx):
        meds.append(np.median(ref_hist[idx]))
    
    med = np.median(meds)
    for chromo in chromo_limits:
      ref_hist = ref_hist_dict[chromo]
      ref_hist_dict[chromo] = ref_hist/med
    
    for row, d2 in enumerate(comp_data_paths):
      ax = axarr[row, col]
      
      mat = _get_anchor_mat(ref_region_dict, ref_hist_dict, track_hist_dicts[d2], chromo_limits, bin_size, n_bins)
      
      if len(mat):
        cax = ax.matshow(mat, cmap=colors, aspect='auto', vmin=0.0, vmax=1.5)
  
      if row == n_comp-1:
        ax.set_xlabel(ref_labels[col], fontsize=9)
        ax.xaxis.set_tick_params(which='both', direction='out', bottom=True, top=False, labeltop=False, labelbottom=True)
        x_bins = np.linspace(0, n_bins, 5)[1:-1]
        ax.set_xticks(x_bins)
        ax.set_xticklabels((x_bins-n_bins/2) * bin_size/int(1e3))
      else:
        ax.set_xticks([])
      
      if row == 0:
        ax.set_title(ref_labels[col], fontsize=9)
      
      if col == 0:
        ax.set_ylabel(comp_labels[row], fontsize=9)
      
      ax.set_yticks([])
        
  fig.text(0.05, 0.5, 'Signal density track', color='#000000',
           horizontalalignment='center',
           verticalalignment='center', rotation=90,
           fontsize=13, transform=fig.transFigure)

  fig.text(0.5, 0.05, 'Separation from anchor site (kb)', color='#000000',
           horizontalalignment='center',
           verticalalignment='center', rotation=0,
           fontsize=13, transform=fig.transFigure)

  cbax1 = fig.add_axes([0.91, 0.6, 0.02, 0.3]) # left, bottom, w, h
  cbar1 = plt.colorbar(cax, cax=cbax1, orientation='vertical')
  
  cbar1.ax.tick_params(labelsize=8)
  cbar1.set_label('$log_{10}(signal+1)$ medians', fontsize=9)

  
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show()
    
  plt.close()  



def _get_seq_seps(ref_region_dict, region_dict):
  
  up_deltas = []
  dn_deltas = []
  
  for chromo in ref_region_dict:
    if chromo not in region_dict:
      continue
    
    if not len(ref_region_dict[chromo]):
      continue

    if not len(region_dict[chromo]):
      continue
      
    ref_middles = ref_region_dict[chromo].mean(axis=1).astype(int)
    middles = np.sort(region_dict[chromo].mean(axis=1)).astype(int)
    n = len(middles)
    
    idx = np.searchsorted(middles, ref_middles)
    
    deltas = middles[np.clip(idx, 0, n-1)] - ref_middles
    deltas_dn = ref_middles - middles[np.clip(idx-1, 0, n-1)] #  downstream
    
    idx = deltas_dn < deltas
    
    dn_deltas.append(deltas_dn[idx])
    up_deltas.append(deltas[~idx])
     
  up_deltas = np.clip(np.concatenate(up_deltas, axis=0), 1, INF)
  dn_deltas = np.clip(np.concatenate(dn_deltas, axis=0), 1, INF)
  
  return up_deltas, dn_deltas


def plot_seq_sep_distribs(raw_data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, pdf, seq_reg_size=None, n_bins=50):

  log = False if seq_reg_size else True
  
  n_ref = len(ref_data_paths)
  n_comp = len(comp_data_paths)
  
  n_rows = int(math.ceil(math.sqrt(n_ref)))
  n_cols = int(math.ceil(n_ref / float(n_rows)))
  
  fig, axarr = plt.subplots(n_rows, n_cols, squeeze=False) # , sharex=True, sharey=True)
    
  plt.subplots_adjust(left=0.12, bottom=0.15, right=0.85, top=0.9, wspace=0.15, hspace=0.2)

  fig.text(0.5, 0.95, 'Separations to closest sites', color='#000000',
          horizontalalignment='center',
          verticalalignment='center',
          fontsize=13, transform=fig.transFigure)
  
  fig.set_size_inches(3*n_cols, 2*n_rows)
  
  if log:
    x_label = 'Seq. separation $log_{10}(bp)$'
  
  else:
    x_label = 'Seq. separation $(kb)$'
    
  used = np.zeros((n_rows, n_cols))
  
  if log:
    x_range = (-7.0, 7.0)
  else:
    x_range = (-seq_reg_size/2.0, seq_reg_size/2.0)
  
  xl, xu = x_range
  y_max_all = 0.0
  used_labels = set()
  
  for i, d1 in enumerate(ref_data_paths):
    row = int(i//n_cols)
    col = i % n_cols
    
    used[row, col] = 1
        
    ax = axarr[row, col]
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    
    ax.set_xlim(*x_range) 
    
    num_sites = sum(len(raw_data_dict[d1][0][c]) for c in raw_data_dict[d1][0])
    y_max = 0.0
    
    if i == 0:
      for j, d2 in enumerate(comp_data_paths):
        color = PLOT_CMAP(j/float(n_comp-1))
        label = comp_labels[j]
        ax.plot([], alpha=0.65, color=color, label=label)
        
    for j, d2 in enumerate(comp_data_paths):
      
      if j == i:
        continue
        
      color = PLOT_CMAP(j/float(n_comp-1))
      
      up_seq_seps, dn_seq_seps = _get_seq_seps(raw_data_dict[d1][0], raw_data_dict[d2][0])
      
      if log:
        up_seq_seps = np.log10(up_seq_seps)
        dn_seq_seps = np.log10(dn_seq_seps)
      else:
        up_seq_seps = up_seq_seps.astype(float)/1e3
        dn_seq_seps = dn_seq_seps.astype(float)/1e3
              
      up_hist, edges = np.histogram(up_seq_seps, bins=n_bins, range=(0, xu))
      dn_hist, edges = np.histogram(dn_seq_seps, bins=n_bins, range=(0, -xl))
      
      hist = np.hstack([dn_hist[::-1], up_hist]).astype(float)
      hist *= 100.0/float(num_sites)
       
      y_max = max(y_max, hist.max())
      
      x_vals = np.linspace(xl, xu, n_bins*2)

      ax.plot(x_vals, hist, linewidth=0.5, alpha=0.65, color=color, zorder=j+n_comp)
            
    y_max_all = max(y_max_all, y_max)
    ax.set_title('{}  $n={:,}$'.format(ref_labels[i], num_sites), fontsize=8)
   
  fig.legend(frameon=False, fontsize=9, loc=5, ncol=1)
  
  for col in range(n_cols):
    lr = used[:,col].sum()-1
    
    for row in range(n_rows):
      ax = axarr[row, col]
      ax.grid(True, which='major', axis='x', alpha=0.5, linestyle='--')
      ax.set_ylim(-0.05*y_max_all, 1.1*y_max_all)
      
      if log:
        ax.set_ylim(-0.05*y_max_all, 1.1*y_max_all)
      
      if row < lr:
        ax.set_xticklabels([])
         
      elif row == lr:
        ax.set_xlabel(x_label, fontsize=8)
        
        if log:
          xticks = np.arange(xl+1,xu,2)
          xlabels = ['$-10^{%d}$' % x for x in np.arange(-xl-1,1,-2)] + ['0'] + ['$10^{%d}$' % x for x in np.arange(2,xu,2)]
          ax.set_xticks(xticks)
          ax.set_xticklabels(xlabels)
          
        #else:
        #  xticks = np.arange(xl/1e3,1+xu/1e3,2e3)
        #  ax.set_xticks(xticks)
      
      else:
         ax.axis('off')

  fig.text(0.05, 0.5, 'Abundance (% reference sites)', color='#000000',
           horizontalalignment='center',
           verticalalignment='center', rotation=90,
           fontsize=13, transform=fig.transFigure)
  
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show()
    
  plt.close()  


def data_track_compare(ref_data_paths, comp_data_paths, ref_labels, comp_labels, out_pdf_path,
                       bin_size=DEFAULT_BIN_KB, seq_reg_size=DEFAULT_SEQ_REG_KB, colors='Blues', screen_gfx=False):
  
  from nuc_tools import io, util
  
  ref_labels = io.check_file_labels(ref_labels, ref_data_paths)
  
  if comp_data_paths:
    io.check_file_labels(comp_labels, comp_data_paths)
  
  else:
    comp_data_paths = ref_data_paths
    comp_labels = ref_labels

  if out_pdf_path:
    out_pdf_path = io.check_file_ext(out_pdf_path, '.pdf')
  
  else:
    out_pdf_path = io.get_out_job_file_path(ref_data_paths[0], DEFAULT_PDF_OUT, (len(ref_data_paths), len(comp_data_paths)))
  
  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(out_pdf_path)    
  
  bin_size *= 1e3
  
  data_dict, raw_data_dict, chromo_limits = _load_bin_data(ref_data_paths+comp_data_paths, bin_size)
  
  plot_value_distribs(raw_data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, pdf)

  plot_seq_anchor_mat(raw_data_dict, chromo_limits, ref_data_paths, comp_data_paths, ref_labels, comp_labels, colors, pdf, seq_reg_size)
  
  plot_seq_sep_distribs(raw_data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, pdf, seq_reg_size)

  plot_seq_sep_distribs(raw_data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, pdf)
  
  plot_data_line_correlations(data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, pdf)
      
  #plot_data_correlations('correlation', data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, colors, pdf)
  
  #plot_data_correlations('semi-quantile', data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, colors, pdf, quantile_x=True)

  plot_data_correlations('quantile bin violin', data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, colors, pdf, quantile_y=False, violin=True)

  #plot_data_correlations('quantile bin box', data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, colors, pdf, quantile_x=True, boxplot=True)
  
  if pdf:
    pdf.close()
    util.info('Written {}'.format(out_pdf_path))
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
  
  arg_parse.add_argument(metavar='BED_FILES', nargs='+', dest='d',
                         help='Primary input data track files to be compared, in BED format. All data tracks will be compared to all others inless the -d is used.')

  arg_parse.add_argument('-l', '--data-labels', metavar='DATA_NAMES', nargs='+', dest='l',
                         help='Optional textual labels/names for the primary input data tracks.')

  arg_parse.add_argument('-d2', '--data-files2', metavar='BED_FILES', nargs='+', dest='d2',
                         help='Secondary input data track files to be compared, in BED format. All primary data will be compared to all secondary data.')

  arg_parse.add_argument('-l2', '--data-labels2', metavar='DATA_NAMES', nargs='+', dest='l2',
                         help='Optional textual labels/names for the secondary input data tracks.')
  
  out_example = DEFAULT_PDF_OUT.format('{#}','{#}', '{#}')
  arg_parse.add_argument('-o', '--out-pdf', metavar='OUT_PDF_FILE', default=None, dest='o',
                         help='Optional output PDF file name. If not specified, a default of the form %s will be used.' % out_example)

  arg_parse.add_argument('-s', '--bin-size', default=DEFAULT_BIN_KB, metavar='BIN_SIZE', type=float, dest="s",
                         help='Binned sequence region size, in kilobases, for comparing data tracks are across equal sized chromosome regions.' \
                              'Default is {:.1f} kb .'.format(DEFAULT_BIN_KB))

  arg_parse.add_argument('-w', '--seq-width', default=DEFAULT_SEQ_REG_KB, metavar='REGION_SIZE', type=float, dest="w",
                         help='Analysis region width, in kilobases, used in sequence separation plots.' \
                              'Default is {:.1f} kb .'.format(DEFAULT_SEQ_REG_KB))

  arg_parse.add_argument('-g', '--screen-gfx', default=False, action='store_true', dest='g',
                         help='Display graphics on-screen using matplotlib, where possible and ' \
                              'do not automatically save graphical output to file.')

  arg_parse.add_argument('-colors', metavar='COLOR_SCALE', default='#FFFFFF,#4080FF,#000000',
                         help='Optional scale colours as a comma-separated list, e.g. "white,blue,red".' \
                              'or colormap (scheme) name, as used by matplotlib. ' \
                              'Note: #RGB style hex colours must be quoted e.g. "#FF0000,#0000FF" ' \
                              'See: %s This option overrides -b.' % util.COLORMAP_URL)

  args = vars(arg_parse.parse_args(argv))
                                
  ref_data_paths = args['d']
  comp_data_paths = args['d2'] or []
  ref_labels = args['l'] or []
  comp_labels = args['l2'] or []
  out_pdf_path = args['o']
  bin_size = args['s']
  seq_reg_size = args['w']
  screen_gfx = args['g']
  colors = args['colors']
  
  for in_path in ref_data_paths + comp_data_paths:
    io.check_invalid_file(in_path)
    
  if colors:
    colors = util.string_to_colormap(colors) 
  
  if len(ref_data_paths + comp_data_paths) < 2:
    util.critical('A total of at least two datasets are required for comparison')
      
  data_track_compare(ref_data_paths, comp_data_paths, ref_labels, comp_labels, out_pdf_path, bin_size, seq_reg_size, colors, screen_gfx)
  
if __name__ == "__main__":
  sys.path.append(dirname(dirname(__file__)))
  main()

"""
TBC
---
Density matrix of all track similarities
 - correlations
 - sequence proximity

Plot for optimal bin size
 - for each pair correlation vs bin_size
   + combine smaller bins for speed
   + log scale for dynamic range
 
./nuc_tools data_track_compare /data/bed/H3K4me3_hap_EDL.bed /data/bed/H3K9me3_hap_EDL.bed -o test.pdf -d2 /data/bed/H3K27me3_hap_EDL.bed /data/bed/H3K27ac_GEO.bed /data/bed/H3K36me3_hap_EDL.bed /data/bed/CTCF_hap_EDL.bed /data/bed/Smc3_hap_EDL.bed -l H3K4me3 H3K9me3 -l2 H3K27me3 H3K27ac H3K36me3 CTCF Smc3
./nuc_tools data_track_compare /data/bed/H3K4me3_hap_EDL.bed /data/bed/H3K9me3_hap_EDL.bed /data/bed/H3K27me3_hap_EDL.bed /data/bed/H3K27ac_GEO.bed /data/bed/H3K36me3_hap_EDL.bed /data/bed/CTCF_hap_EDL.bed /data/bed/Smc3_hap_EDL.bed -o /home/tjs23/Desktop/test_dt_comp.pdf -l H3K4me3 H3K9me3 H3K27me3 H3K27ac H3K36me3 CTCF Smc3
"""

