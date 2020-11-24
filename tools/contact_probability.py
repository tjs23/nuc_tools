import sys,  os, math
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.backends.backend_pdf import PdfPages

PROG_NAME = 'contact_probability'
VERSION = '1.0.1'
DESCRIPTION = 'Chromatin contact (NCC or NPZ format) probability vs sequence separation and region analysis'
DEFAULT_BIN_SIZE = 25
REGION_TILES = ('Within','Overlap','Outside')
PLOT_CMAP = LinearSegmentedColormap.from_list(name='PLOT_CMAP', colors=['#00C0C0','#0040FF','#FF0000','#C0C000','#808080'], N=255)   
HIST_CMAP = LinearSegmentedColormap.from_list(name='HIST_CMAP', colors=['#FFFFFF','#0080FF','#FF0000', '#FFFF00'], N=255)   
PDF_DPI = 200

def load_seq_seps(contact_paths, labels, region_dict, bin_size):

  from nuc_tools import util, io
  from formats import ncc, npz, bed
  
  seq_sep_data = []
  
  for i, in_path in enumerate(contact_paths):
    util.info('Processing %s (%s)' % (in_path, labels[i]))
    seq_seps = []
    weights = []
    counts = []
    seq_seps_r = None
    weights_r = None
    
    if region_dict:
      seq_seps_r = [[], [], []]
      weights_r = [[], [], []]
    
    if io.is_ncc(in_path):
      file_bin_size = None
      util.info('  .. loading')
      chromosomes, chromo_limits, contacts = ncc.load_file(in_path, trans=False, dtype=np.int32)
      chromosomes = util.sort_chromosomes(chromosomes)
      
    else:
      util.info('  .. loading')
      file_bin_size, chromo_limits, contacts = npz.load_npz_contacts(in_path, trans=False)
      
      if file_bin_size > bin_size:
        util.critical('Binned resolution of file (%d kb) is greater than analysis bin size (%d kb)' % (file_bin_size/1e3, bin_size/1e3))
      
      chromosomes = chromo_limits.keys()
    
    chromosomes = util.sort_chromosomes(chromosomes)
    
    for chr_a in chromosomes:
      chromo_pair = chr_a, chr_a
      
      if chromo_pair not in contacts:
        continue
      
      util.info('  .. %s' % chr_a, line_return=True)
      
      if file_bin_size:
        matrix = contacts[chromo_pair]
        n = len(matrix)
        prob = []
        seps = []
        seq_pos_a = []
        seq_pos_b = []
        chromo_counts = []
        
        for d in range(0, n):
          m = n-d
          rows = np.array(range(m))
          cols = rows + d
          idx = (rows, cols)
 
          frac = n/float(m)
          ss = np.full(m, d * file_bin_size)
          ct = matrix[idx]
          nz = ct.nonzero()
          
          if len(nz[0]):
            ss = ss[nz]
            ct = ct[nz]
            chromo_counts.append(ct)
            
            pa = chromo_limits[chr_a][0] + rows * file_bin_size
            pb = chromo_limits[chr_a][0] + cols * file_bin_size
            
            prob.append(ct*frac) # - weights come from fraction of chromo and sum of counts
            seps.append(ss)
            seq_pos_a.append(pa[nz])
            seq_pos_b.append(pb[nz])
        
        chromo_counts = np.concatenate(chromo_counts)
        seps = np.concatenate(seps)
        prob = np.concatenate(prob)
        seq_pos_a = np.concatenate(seq_pos_a)
        seq_pos_b = np.concatenate(seq_pos_b)
       
      else:
        contact_array = contacts[chromo_pair]
        seq_pos_a = contact_array[:,0]
        seq_pos_b = contact_array[:,1]
        chromo_counts = contact_array[:,2]
        
        seps = abs(seq_pos_a-seq_pos_b)
        indices = seps.nonzero()
        seps = seps[indices]

        p_start, p_end = chromo_limits[chr_a]
        size = float(p_end-p_start+1)

        prob = (size/(size-seps)) # From fraction of chromosome that could give rise to each separation
        seps = seps
       
      if len(prob):
        counts.append(chromo_counts)
        seq_seps.append(seps)
        weights.append(prob)
        
        if region_dict:
          regions = region_dict[chr_a]
 
          if not len(regions):
            continue
         
          rstarts = regions[:,0]
          rends = regions[:,1]
          rstarts = np.append(rstarts, [rends[-1]])
          
          e1 = np.searchsorted(rends, seq_pos_a) # Region indices for each contact
          e2 = np.searchsorted(rends, seq_pos_b)
          
          # Are seq pos at or above the region starts corresponding to the region ends that they are immediately less than
          in_regions_a = seq_pos_a >= rstarts[e1]
          in_regions_b = seq_pos_b >= rstarts[e2]
         
          intra = in_regions_a & in_regions_b # Elements where both pos are in any region
          inter = np.logical_xor(in_regions_a, in_regions_b) # Only one pos is in a region
          extra = ~(intra | inter) # Neither pos in a region
          
          
          for r, idx in enumerate((intra, inter, extra)):
            w = prob[idx]
            
            if len(w):
              seq_seps_r[r].append(seps[idx])
              weights_r[r].append(w)
    
    counts = np.concatenate(counts)
    seq_seps = np.concatenate(seq_seps)
    weights = np.concatenate(weights)
    
    seq_sep_data.append((seq_seps, weights, seq_seps_r, weights_r, counts))
    
    util.info('  .. found {:,} values'.format(len(seq_seps)))
    
  return seq_sep_data
  
  

def plot_seq_sep_distrib(seq_sep_data, labels, region_label, bin_size, max_sep=1e8, pdf=None):

  n_cont = len(seq_sep_data)
  colors = [PLOT_CMAP(x) for x in np.linspace(0.0, float(n_cont), n_cont)]
  bin_size *= 1e3
  
  if seq_sep_data[0][2]:
    fig, axarr = plt.subplots(2,2)
    ax0 = axarr[0,0]
    ax1 = axarr[0,1]
    ax2 = axarr[1,0]
    ax3 = axarr[1,1]
    axes = (ax0, ax1, ax2, ax3)
    
  else:
    fig, ax0 = plt.subplots()
    axes = [ax0]
 
  fig.set_size_inches(7, 7)    
  plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.9, wspace=0.1, hspace=0.1)

  x_limit = max_sep
  num_bins = (x_limit-bin_size)/bin_size
  bins = np.linspace(2*bin_size, x_limit, num_bins)

  y_mins = []
  y_maxs = []
   
  for i, (seq_seps, weights, seq_seps_r, weights_r, c) in enumerate(seq_sep_data):
    hist, edges = np.histogram(seq_seps, bins=bins, weights=weights, density=True)

    idx = hist.nonzero()

    hist = hist[idx]
    edges = edges[idx]

    x_data = np.log10(edges)
    y_data = np.log10(hist)

    y_mins.append(y_data.min())
    y_maxs.append(y_data.max())
    
    ax0.plot(x_data, y_data, label=labels[i], color=colors[i], linewidth=1, alpha=0.5)
    
    # For each bin-pair, there is a seq-sep bin and a bin-count
    # - look at distribs of counts in each bin
    
    if seq_seps_r:
      for r, ax in enumerate((ax1, ax2, ax3)):
        hist, edges = np.histogram(np.concatenate(seq_seps_r[r]), bins=bins, weights=np.concatenate(weights_r[r]), density=True)

        idx = hist.nonzero()

        hist = hist[idx]
        edges = edges[idx]

        x_data = np.log10(edges)
        y_data = np.log10(hist)
        
        ax.set_title('%s %s regions' % (REGION_TILES[r], region_label))
        ax.plot(x_data, y_data, label=labels[i], color=colors[i], linewidth=1, alpha=0.5)
 
  x_min = 0.5 * int(math. ceil(2.0 * np.log10(bin_size)))
  x_range = np.arange(x_min, np.log10(x_limit), 0.5)
  y_min = int(2.0 * min(y_mins))/2.0 - 0.5
  y_max = int(2.0 * max(y_maxs))/2.0 + 0.5
  y_range = np.arange(y_min, y_max, 0.5)
 
  x1 = x_min + 0.5
  x2 = x1 + 1.5
  y1 = y_max-0.5
  y2 = y1 - 1.0
  v1 = y1 - (x2-x1)
  v2 = y2 - 1.5 * (x2-x1)
  
  ax0.set_title('Contact sequence separations')
  
  for ax in axes:
    ax.set_alpha(0.5)

    ax.set_xlabel('Sequence separation (bp)')
    ax.set_ylabel('Contact probability (%d kb bins)' % (bin_size/1e3))
    ax.xaxis.set_ticks(x_range)
    ax.set_xticklabels(['$10^{%.1f}$' % x for x in x_range], fontsize=10)
    ax.set_xlim((x_min, np.log10(x_limit)))
 
    ax.yaxis.set_ticks(y_range)
    ax.set_yticklabels(['$10^{%.1f}$' % x for x in y_range], fontsize=10)
    ax.set_ylim((y_min, y_max))
    
    ax.plot([x1, x2], [y1, v1], color='#808080', alpha=0.5, linestyle='--')
    ax.plot([x1, x2], [y2, v2], color='#808080', alpha=0.5, linestyle='--')
    ax.text(x2, v1, '$\lambda=1.0$', color='#808080', verticalalignment='center', alpha=0.5, fontsize=14)
    ax.text(x2, v2, '$\lambda=1.5$', color='#808080', verticalalignment='center', alpha=0.5, fontsize=14)

    ax.legend()

  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show()
    
  plt.close()    


def plot_count_distribs(seq_sep_data, labels, bin_size, pdf):

  n_plots = len(seq_sep_data)

  n_rows = int(math.ceil(math.sqrt(n_plots)))
  n_cols = int(math.ceil(n_plots / float(n_rows)))
  
  fig, axarr = plt.subplots(n_rows, n_cols, squeeze=False) # , sharex=True, sharey=True)
    
  plt.subplots_adjust(left=0.1, bottom=0.15, right=0.85, top=0.87, wspace=0.25, hspace=0.1)

  fig.text(0.5, 0.95, 'Hi-C count distributions ({:,} kb bins)'.format(bin_size), color='#000000',
          horizontalalignment='center',
          verticalalignment='center',
          fontsize=13, transform=fig.transFigure)
  
  fig.set_size_inches(max(5, 4*n_cols), max(5, 4*n_rows)) 
  
  x_label = 'Seq. separation (bp)'
  y_label = '$log_{10}(bin count)$'
  
  x_min = int(np.log10(bin_size*1e3))
  
  x_max = int(math.ceil(np.log10(max([d[0].max() for d in seq_sep_data]))))
  y_min = 0
  y_max = np.log10(max([d[4].max() for d in seq_sep_data]))
  
  x_bins = 50
  y_bins = 50
  
  nx_ticks = int(x_max-x_min) + 1
  x_ticks = np.linspace(0, x_bins, nx_ticks)
  x_labels = ['$10^{%d}$' % x for x in np.linspace(x_min, x_max, nx_ticks)]
  
  y_vals = np.arange(y_min, y_max+0.5, 0.5)
  y_ticks = np.linspace(0, y_bins*(y_vals[-1]/float(y_max)), len(y_vals))
  
  y_labels = ['$10^{%.1f}$' % x for x in y_vals]
  
  x_bins = np.logspace(x_min, x_max, x_bins)
  #x_bins = np.linspace(0, x_bins*bin_size*1e3, x_bins)
  y_bins = np.logspace(y_min, y_max, y_bins)
   
  for i,(ss, wt, ss2, wt2, counts) in enumerate(seq_sep_data):
    row = int(i//n_cols)
    col = i % n_cols
  
    ax = axarr[row, col]
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
        
    hist, e1, e2 = np.histogram2d(ss, counts, bins=(x_bins, y_bins))
    
    hist = hist.T
    
    nz = hist > 0
    
    hist[nz] = np.log10(hist[nz])
    
    hist[nz] /= hist.max()
    
    cax = ax.imshow(hist, cmap=HIST_CMAP, origin='lower')
    
    #for j, d in enumerate(hist):
    #  d = d[d>0]
    #  ax.plot(d, color=HIST_CMAP(j/float(len(hist))), linewidth=1, alpha=0.5)
    
    #ax.scatter(np.log10(ss), np.log10(counts), s=2, alpha=0.1)
   
    if col == 0:
      ax.set_ylabel(y_label, fontsize=9)
      ax.set_yticks(y_ticks)
      ax.set_yticklabels(y_labels) 
         
    ax.set_title(labels[i], fontsize=9)
    
    if row == n_rows-1:
      ax.set_xticks(x_ticks)
      ax.set_xticklabels(x_labels)
      ax.set_xlabel(x_label, fontsize=9)
    else:
      ax.set_xticks([])
  
  i += 1
  while i < (n_rows*n_cols):
    row = int(i//n_cols)
    col = i % n_cols
    axarr[row, col].axis('off')
    
    axarr[row-1, col].set_xticks(x_ticks)
    axarr[row-1, col].set_xticklabels(x_labels)
    axarr[row-1, col].set_xlabel(x_label, fontsize=9)
    
    i += 1
  
  
  cbax1 = fig.add_axes([0.87, 0.6, 0.02, 0.3]) # left, bottom, w, h
  cbar1 = plt.colorbar(cax, cax=cbax1, orientation='vertical')
  cbar1.ax.tick_params(labelsize=8)
  cbar1.set_label('Probability/maximum', fontsize=9)

  
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show()
    
  plt.close()    


def contact_probability(contact_paths, out_pdf_path=None, region_path=None, bin_size=DEFAULT_BIN_SIZE,
                        labels=None, region_label=None, screen_gfx=False):

  from nuc_tools import util, io
  from formats import ncc, npz, bed
  
  if not out_pdf_path:
    out_pdf_path = os.path.splitext(contact_paths[0])[0] + '_contact_prob.pdf' 
    
  out_pdf_path = io.check_file_ext(out_pdf_path, '.pdf')
  
  if labels:
    for i, label in enumerate(labels):
      labels[i] = label.replace('_',' ')
      
    while len(labels) < len(contact_paths):
      labels.append(os.path.basename(contact_paths[len(labels)]))
      
  else:
    labels = [os.path.splitext(os.path.basename(x))[0] for x in contact_paths]
  
  if region_path:
    region_dict, value_dict, label_dict = bed.load_bed_data_track(region_path)
    
    for chr_a in region_dict:
      ends = region_dict[chr_a][:,0]
      idx = ends.argsort()
      region_dict[chr_a] = region_dict[chr_a][idx,:]
    
    if not region_label:
      region_label = os.path.basename(region_path)
    
  else:
    region_dict = None

  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(out_pdf_path) 
    
  seq_sep_data = load_seq_seps(contact_paths, labels, region_dict, bin_size*1e3)
    
  plot_seq_sep_distrib(seq_sep_data, labels, region_label,
                       bin_size, pdf=pdf)     
  
  plot_count_distribs(seq_sep_data, labels, bin_size, pdf)  
  
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

  arg_parse.add_argument(metavar='CONTACT_FILES', nargs='+', dest='i',
                         help='One or more input NPZ or NCC format chromatin contact file(s). Wildcards accepted')

  arg_parse.add_argument('-o', '--out-pdf', metavar='PDF_FILE', dest='o',
                         help='Output PDF format file. If not specified, a default based on the input file name(s).')

  arg_parse.add_argument('-g', '--gfx', default=False, action='store_true', dest='g',
                         help='Display graphics on-screen using matplotlib, where possible and do not automatically save output.')

  arg_parse.add_argument('-l', '--labels', metavar='LABELS', nargs='*', dest="l",
                         help='Text labels for the input files (otherwise the input file names wil be used)')

  arg_parse.add_argument('-r', '--regions', metavar='REGION_FILE', dest="r",
                         help='Data track file in BED format specifying chromosome analysis regions')

  arg_parse.add_argument('-rl', '--regions_label', metavar='LABEL', dest="rl",
                         help='Optional text label for the input regions, where specified.')

  arg_parse.add_argument('-s', '--bin-size', default=DEFAULT_BIN_SIZE, metavar='KB_BIN_SIZE', type=int, dest='s',
                         help='The sequence region size in kilobases for calculation of contact probabilities. ' \
                              'Cannot be smaller than for any pre-binned contact files.' \
                              'Default is %d (kb)' % DEFAULT_BIN_SIZE)
 
  args = vars(arg_parse.parse_args(argv))

  contact_paths = args['i']
  region_path = args['r']
  screen_gfx  = args['g']  
  pdf_path = args['o']
  bin_size = args['s']
  labels = args['l'] or None
  region_label = args['rl']
  
  if region_path:
    io.check_invalid_file(region_path)
  
  for file_path in contact_paths:
    io.check_invalid_file(file_path)
   
  if pdf_path and screen_gfx:
    util.warn('Output PDF file will not be written in screen graphics (-g) mode')
    pdf_path = None
  
  contact_probability(contact_paths, pdf_path, region_path, bin_size, labels, region_label, screen_gfx)
  
  
  

if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()


  """
./nuc_tools contact_probability /data/dino_hi-c/SLX-17943_NoIndex_HCJ37DRXX_s_2_r_1_2_5k.npz -o test.pdf -s 5
./nuc_tools contact_probability /data/SORT_ME/hi-c/Frezza_MRCU/Hi-C_SLX-12296_100k.npz /data/SORT_ME/hi-c/Frezza_MRCU/Hi-C_SLX-14982_100k.npz -s 100 -o test.pdf
./nuc_tools contact_probability /data/SORT_ME/hi-c/pop/SLX-*25k.npz  -s 25 -o test.pdf
  """
