import os, sys, math
import numpy as np
from glob import glob
from os.path import dirname

PROG_NAME = 'data_track_compare'
VERSION = '1.0.0'
DESCRIPTION = 'Plot and measure similarities between data tracks in BED format'

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Colormap

DEFAULT_PDF_OUT = 'dtc_out_job{}_D{}x{}.pdf'
DEFAULT_BIN_KB  = 20.0
PDF_DPI = 200

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
  
  util.info('Binned data tracks into {:,} regions'.format(len(data_dict[data_bed_path])))
                                                   
  return binned_data_dict, data_dict
  
HIST_BINS2D = 50

LOG_THRESH = 4

NUM_QUANT_BINS = 10

COLORS = ['#FF8080','#FFC060','#60C0F0','#808080','#80E080','#E080E0','#80E0E0']

def plot_data_line_correlations(data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, pdf):

  from scipy import stats
  
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
  
  for i, d1 in enumerate(ref_data_paths):
    row = int(i//n_cols)
    col = i % n_cols
    
    vals1 = data_dict[d1]
    nzy = vals1 > 0

    mn, md, mx  = np.percentile(vals1[nzy], [0.5, 50.0, 99.5])    
    #y_lim = (np.log10(mn), np.log10(mx))
 
    ax = axarr[row, col]
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    #ax.set_ylim(*y_lim)
    
    x_lim = (0.0, 100.0)
    
    
    
    for j, d2 in enumerate(comp_data_paths):
      if j == i:
        continue
    
      vals2 = data_dict[d2]
      nzx = vals2 > 0
      nz = nzx & nzy
      
      y_vals = vals1[nz]
      x_vals = vals2[nz]
      
      y_vals = np.log10(y_vals)
      
      x_vals = stats.rankdata(x_vals)
      x_vals /= float(len(x_vals))
      idx = x_vals.argsort()
      x_vals = x_vals[idx]
      y_vals = y_vals[idx]
 
      bw = 100/nq
 
      y_split = np.array_split(y_vals, nq)
 
      x_pos = np.arange(0, nq)
      
      y_med = np.array([np.mean(d) for d in y_split])
      
      y_sem = np.array([stats.sem(d) for d in y_split])
      
      #y_q25 = [np.percentile(d, 25.0) for d in y_split]
      #y_q75 = [np.percentile(d, 75.0) for d in y_split]
      
      color = COLORS[j % len(COLORS)]
      
      #ax.plot(x_pos, y_med, label=comp_labels[col], alpha=0.5)

      ax.errorbar(x_pos, y_med, y_sem, linewidth=1, alpha=0.67, color=color, label=comp_labels[j], zorder=j+n_comp)
      
      #ax.fill_between(x_pos, y_q25, y_q75, linewidth=0, alpha=0.2, color=color, zorder=j)
      
      
    ax.set_xlim((0, nq)) 
    
    ax.set_ylabel(ref_labels[i], fontsize=9)
    
    if row == n_rows-1:
      ax.set_xticks(np.arange(0, nq+1)-0.5)
      ax.set_xticklabels(bw*np.arange(0, nq+1))
      ax.set_xlabel(x_label, fontsize=9)
    else:
      ax.set_xticks([])
   
  fig.legend(frameon=False, fontsize=9, loc=8, ncol=int((n_ref+1)/2))
  
  i += 1
  while i < (n_rows*n_cols):
    row = int(i//n_cols)
    col = i % n_cols
    axarr[row, col].axis('off')
    
    axarr[row-1, col].set_xticks(np.arange(0, nq+1)-0.5)
    axarr[row-1, col].set_xticklabels(bw*np.arange(0, nq+1))
    axarr[row-1, col].set_xlabel(x_label, fontsize=9)
    
    i += 1

  
  fig.text(0.05, 0.5, 'Mean $log_{10}$(Data value)', color='#000000',
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
  
  from scipy import stats
  
  n_ref = len(ref_data_paths)
  n_comp = len(comp_data_paths)
  
  fig, axarr = plt.subplots(n_ref, n_comp, squeeze=False) # , sharex=True, sharey=True)
  
  fig.set_size_inches(max(8, 2*n_comp), max(8, 2*n_ref))
  
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.08, hspace=0.08)

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
      r, p = stats.pearsonr(x_vals, y_vals)
      rho, p = stats.spearmanr(x_vals, y_vals)
      
      if log_y:
        y_vals = np.log10(y_vals)
      elif quantile_y:
        y_vals = stats.rankdata(y_vals)
        y_vals /= float(len(y_vals))

      if violin or boxplot:
        x_vals = stats.rankdata(x_vals)
        x_vals /= float(len(x_vals))
        idx = x_vals.argsort()
        x_vals = x_vals[idx]
        y_vals = y_vals[idx]
        
        bw = 100/NUM_QUANT_BINS
         
        y_split = np.array_split(y_vals, NUM_QUANT_BINS)
        
        if violin:
          x_pos = np.arange(0, NUM_QUANT_BINS)
          vp = ax.violinplot(y_split, x_pos, points=50, widths=0.8, bw_method=0.25,
                           showmeans=False, showextrema=False, showmedians=True)
          ax.set_xlim((-1, NUM_QUANT_BINS))
          ax.set_xticks(np.arange(0, NUM_QUANT_BINS+1)-0.5)
          ax.set_xticklabels(bw*np.arange(0, NUM_QUANT_BINS+1), rotation=90.0)
          
          color = COLORS[col % len(COLORS)]
          for i, a in enumerate(vp['bodies']):
            a.set_facecolor(color)
            a.set_edgecolor('black')
 
          vp['cmedians'].set_color('black')
          
        else:
          color = COLORS[col % len(COLORS)]
          bp = ax.boxplot(y_split, notch=True, sym=',', widths=0.8, whis=[10,90],
                          boxprops={'color':color},
                          capprops={'color':color},
                          whiskerprops={'color':color},
                          flierprops={'color':color, 'markeredgecolor':color, },
                          medianprops={'color':'#000000'})
        
          ax.set_xlim((0, NUM_QUANT_BINS+1))
          ax.set_xticks(0.5 + np.arange(0, NUM_QUANT_BINS+1))
          ax.set_xticklabels(bw*np.arange(0, NUM_QUANT_BINS+1), rotation=90.0)
                   
      else:

        if log_x:
          x_vals = np.log10(x_vals)
        elif quantile_x:
          x_vals = 100 * stats.rankdata(x_vals)
          x_vals /= float(len(x_vals))
          
        ax.hist2d(x_vals, y_vals,
                  bins=HIST_BINS2D, range=(x_lim, y_lim),
                  cmap=colors)
 
       
        ax.set_xlim(*x_lim)
        
      ax.set_ylim(*y_lim) 
       
      if row == n_ref-1:
        ax.set_xlabel(comp_labels[col], fontsize=9)
      else:
        ax.set_xticks([])
      
      if col == 0:
        ax.set_ylabel(ref_labels[row], fontsize=9)
      else:
        ax.set_yticks([])
        
      ax.text(0.05, 0.95, 'n={:,}\nr={:.2f}\n$\\rho$={:.2f}'.format(n,r,rho),
              color='#202020', verticalalignment='top',
              alpha=1.0, fontsize=8, transform=ax.transAxes)
 

  fig.text(0.5, 0.05, fig_xlabel, color='#000000',
           horizontalalignment='center',
           verticalalignment='center',
           fontsize=13, transform=fig.transFigure)

  fig.text(0.05, 0.5, fig_ylabel, color='#000000',
           horizontalalignment='center',
           verticalalignment='center', rotation=90,
           fontsize=13, transform=fig.transFigure)
 
 
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show()
    
  plt.close()  


def plot_seq_sep_distribs(raw_data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, pdf)

  n_ref = len(ref_data_paths)
  n_comp = len(comp_data_paths)
  
  n_rows = int(math.ceil(math.sqrt(n_ref)))
  n_cols = int(math.ceil(n_ref / float(n_rows)))
  
  fig, axarr = plt.subplots(n_rows, n_cols, squeeze=False) # , sharex=True, sharey=True)
    
  plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)

  fig.text(0.5, 0.95, 'Data sequence separation distributions ({:,} kb bins)'.format(bin_size/1e3), color='#000000',
          horizontalalignment='center',
          verticalalignment='center',
          fontsize=13, transform=fig.transFigure)
  
  fig.set_size_inches(3*n_cols, 2*n_rows)
  
  x_label = 'Separation from site'
  
  
  
  for i, d1 in enumerate(ref_data_paths):
    row = int(i//n_cols)
    col = i % n_cols
    
    vals1 = data_dict[d1]
    nzy = vals1 > 0

    mn, md, mx  = np.percentile(vals1[nzy], [0.5, 50.0, 99.5])    
    
    ax = axarr[row, col]
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    
    x_lim = (0.0, 100.0)  
    
    for j, d2 in enumerate(comp_data_paths):
      if j == i:
        continue
    
      vals2 = data_dict[d2]
      nzx = vals2 > 0
      nz = nzx & nzy
      
      y_vals = vals1[nz]
      x_vals = vals2[nz]
      
      y_vals = np.log10(y_vals)
      
      x_vals = stats.rankdata(x_vals)
      x_vals /= float(len(x_vals))
      idx = x_vals.argsort()
      x_vals = x_vals[idx]
      y_vals = y_vals[idx]
 
      bw = 100/nq
 
      y_split = np.array_split(y_vals, nq)
 
      x_pos = np.arange(0, nq)
      
      y_med = np.array([np.mean(d) for d in y_split])
      
      y_sem = np.array([stats.sem(d) for d in y_split])
      
      color = COLORS[j % len(COLORS)]
      

      ax.errorbar(x_pos, y_med, y_sem, linewidth=1, alpha=0.67, color=color, label=comp_labels[j], zorder=j+n_comp)
            
      
    ax.set_xlim((0, nq)) 
    
    ax.set_ylabel(ref_labels[i], fontsize=9)
    
    if row == n_rows-1:
      ax.set_xticks(np.arange(0, nq+1)-0.5)
      ax.set_xticklabels(bw*np.arange(0, nq+1))
      ax.set_xlabel(x_label, fontsize=9)
    else:
      ax.set_xticks([])
   
  fig.legend(frameon=False, fontsize=9, loc=8, ncol=int((n_ref+1)/2))
  
  i += 1
  while i < (n_rows*n_cols):
    row = int(i//n_cols)
    col = i % n_cols
    axarr[row, col].axis('off')
    
    axarr[row-1, col].set_xticks(np.arange(0, nq+1)-0.5)
    axarr[row-1, col].set_xticklabels(bw*np.arange(0, nq+1))
    axarr[row-1, col].set_xlabel(x_label, fontsize=9)
    
    i += 1

  
  fig.text(0.05, 0.5, 'Mean $log_{10}$(Data value)', color='#000000',
           horizontalalignment='center',
           verticalalignment='center', rotation=90,
           fontsize=13, transform=fig.transFigure)
  
  

  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show()
    
  plt.close()  

def data_track_compare(ref_data_paths, comp_data_paths, ref_labels, comp_labels, out_pdf_path,
                       bin_size=DEFAULT_BIN_KB, colors='Blues', screen_gfx=False):
  
  from nuc_tools import io, util
  
  if ref_labels:
    for i, label in enumerate(ref_labels):
      ref_labels[i] = label.replace('_',' ')
      
    while len(ref_labels) < len(ref_data_paths):
      i = len(ref_labels)
      ref_labels.append(io.get_file_root(ref_data_paths[i]))
      
  else:
    ref_labels = [io.get_file_root(x) for x in ref_data_paths]
  
  if comp_data_paths:
    if comp_labels:
      for i, label in enumerate(comp_labels):
        comp_labels[i] = label.replace('_',' ')
 
      while len(comp_labels) < len(comp_data_paths):
        i = len(comp_labels)
        comp_labels.append(io.get_file_root(comp_data_paths[i]))
 
    else:
      comp_labels = [io.get_file_root(x) for x in comp_data_paths]
  
  else:
    comp_data_paths = ref_data_paths
    comp_labels = ref_labels

  if out_pdf_path:
    out_pdf_path = io.check_file_ext(out_pdf_path, '.pdf')
  
  else:    
    dir_path = dirname(ref_data_paths[0])
    
    job = 1
    while glob(os.path.join(dir_path, DEFAULT_PDF_OUT.format(job, '*', '*'))):
      job += 1
    
    file_name = DEFAULT_PDF_OUT.format(job, len(ref_data_paths), len(comp_data_paths))
    out_pdf_path = os.path.join(dir_path, file_name)  

  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(out_pdf_path)    
  
  bin_size *= 1e3
  
  data_dict, raw_data_dict = _load_bin_data(ref_data_paths+comp_data_paths, bin_size)

  plot_data_line_correlations(data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, pdf)
       
  #plot_data_correlations('correlation', data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, colors, pdf)
  
  #plot_data_correlations('semi-quantile', data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, colors, pdf, quantile_x=True)

  #plot_data_correlations('quantile bin violin ', data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, colors, pdf, violin=True)

  #plot_data_correlations('quantile bin box', data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, bin_size, colors, pdf, boxplot=True)
  
  plot_seq_sep_distribs(raw_data_dict, ref_data_paths, comp_data_paths, ref_labels, comp_labels, pdf)
  
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
                         help='Binned sequence region size, in kilobases: data tracks are compared across equal sized chromosome regions.' \
                              'Default is {:.1f} kb .'.format(DEFAULT_BIN_KB))

  arg_parse.add_argument('-g', '--screen-gfx', default=False, action='store_true', dest='g',
                         help='Display graphics on-screen using matplotlib, where possible and ' \
                              'do not automatically save graphical output to file.')

  arg_parse.add_argument('-colors', metavar='COLOR_SCALE', default='#FFFFFF,#80D0FF,#D00000',
                         help='Optional scale colours as a comma-separated list, e.g. "white,blue,red".' \
                              'or colormap (scheme) name, as used by matplotlib. ' \
                              'Note: #RGB style hex colours must be quoted e.g. "#FF0000,#0000FF" ' \
                              'See: %s This option overrides -b.' % util.COLORMAP_URL)

  args = vars(arg_parse.parse_args(argv))
                                
  ref_data_paths = args['d']
  comp_data_paths = args['d2'] or []
  ref_labels = args['l']
  comp_labels = args['l2'] or []
  out_pdf_path = args['o']
  bin_size = args['s']
  screen_gfx = args['g']
  colors = args['colors']
  
  for in_path in ref_data_paths + comp_data_paths:
    io.check_invalid_file(in_path)
    
  if colors:
    colors = util.string_to_colormap(colors) 
  
  if len(ref_data_paths + comp_data_paths) < 2:
    util.critical('A total of at least two datasets are required for comparison')
      
  data_track_compare(ref_data_paths, comp_data_paths, ref_labels, comp_labels, out_pdf_path, bin_size, colors, screen_gfx)
  
if __name__ == "__main__":
  sys.path.append(dirname(dirname(__file__)))
  main()

"""

Plot of seq overlap
 - for each primary the distribution of closest secondary
   + both strands
 - stack colour matrix of all secondary signal centred on primary site
 
Density matrix of all track similarities
 - correlations
 - sequence proximity

Plot for optimal bin size
 - for each pair correlation vs bin_size
   + combine smaller bins for speed
   + log scale for dynamic range
 
./nuc_tools data_track_compare /data/bed/H3K4me3_hap_EDL.bed /data/bed/H3K9me3_hap_EDL.bed -o test.pdf -d2 /data/bed/H3K27me3_hap_EDL.bed /data/bed/H3K27ac_GEO.bed /data/bed/H3K36me3_hap_EDL.bed /data/bed/CTCF_hap_EDL.bed /data/bed/Smc3_hap_EDL.bed -l H3K4me3 H3K9me3 -l2 H3K27me3 H3K27ac H3K36me3 CTCF Smc3
./nuc_tools data_track_compare /data/bed/H3K4me3_hap_EDL.bed /data/bed/H3K9me3_hap_EDL.bed /data/bed/H3K27me3_hap_EDL.bed /data/bed/H3K27ac_GEO.bed /data/bed/H3K36me3_hap_EDL.bed /data/bed/CTCF_hap_EDL.bed /data/bed/Smc3_hap_EDL.bed -o test.pdf -l H3K4me3 H3K9me3 H3K27me3 H3K27ac H3K36me3 CTCF Smc3
"""

