import sys, math, os
import numpy as np
from collections import defaultdict
from random import randint
from scipy.stats import sem, norm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, LogNorm

PROG_NAME = 'contact_points'
VERSION = '1.0.0'
DESCRIPTION = 'Paired-point chromatin contact (NPZ format) analysis'
DEFAULT_BIN_SIZE = 100

def poisson_wald(c1, c2, n1=None, n2=None,
                 smaller=True, larger=True,
                 null_ratio=1.0):
  
  assert smaller or larger
  
  if not n1 or not n2:
    exposure_ratio = 1.0
  else:
    exposure_ratio = n2 / n1
    
  r = null_ratio / exposure_ratio
  
  if not (c1 and c2):
    return None, None

  z_stat = (c1 - c2 * r) / math.sqrt(c1 + c2 * r * r)
  
  if smaller and larger: # Two-sided
    p_val = norm.sf(abs(z_stat))*2
  elif smaller:
    p_val = norm.cdf(z_stat)
  else:
    p_val = norm.sf(z_stat)

  return z_stat, p_val
  
  
def contact_points(paired_region_path, contact_paths, pdf_path,
                   bin_size=DEFAULT_BIN_SIZE, labels=None,
                   screen_gfx=False, tsv_path=None, max_sep=4e6):

  from nuc_tools import util, io
  from formats import bed, ncc, npz  
  from contact_compare import normalize_contacts
  
  bin_size *= 1000
  
  if not pdf_path:
    pdf_path = os.path.splitext(paired_region_path)[0] + '_hi-c_points.pdf' 
    
  pdf_path = io.check_file_ext(pdf_path, '.pdf')
  
  if labels:
    for i, label in enumerate(labels):
      labels[i] = label.replace('_',' ')
      
    while len(labels) < len(contact_paths):
      labels.append(os.path.basename(contact_paths[len(labels)]))
  else:
    labels = [os.path.basename(x) for x in contact_paths]
  
  point_dict = defaultdict(list)
  counts_dict = defaultdict(list)
  
  with open(paired_region_path) as file_obj:
    prev_pair_id = None
    prev = None
    
    for line in file_obj:
      chromo, start, end, pair_id, obs, strand = line.split()
      
      if prev_pair_id == pair_id :
        chr2, start2, end2, n_obs = prev
        
        if chr2 > chromo:
          chr_a, chr_b = chromo, chr2
          sa, ea = int(start), int(end)
          sb, eb = start2, end2
        else:
          chr_a, chr_b = chr2, chromo
          sa, ea = start2, end2
          sb, eb = int(start), int(end)
        
        key = (chr_a, chr_b)
        point_dict[key].append(((sa+ea)/2, (sb+eb)/2))
        prev_pair_id = None
        
      else:
        prev = (chromo, int(start), int(end), int(obs))
        prev_pair_id = pair_id
  
  chromos = set()
  for in_path in contact_paths:
    file_chromos = npz.get_chromosomes(in_path)
    
    if chromos:
      chromos &= file_chromos
    else:
      chromos = file_chromos
  
  if not chromos:
    util.critical('No common chromosomes found for input Hi-C datasets')
  
  else:
    chromos = util.sort_chromosomes(chromos)
    util.info('Found {:,} common chromosomes/contigs in Hi-C datasets'.format(len(chromos)))
  
  n_inp = len(contact_paths)
  all_counts = []
  all_deltas = []
  
  from matplotlib import pyplot as plt
  
  ref_mats = {}
  ref_chromo_limits = None
  
  num_loops = 0
  
  for i, in_path in enumerate(contact_paths):
    
    file_bin_size, chromo_limits, contacts = npz.load_npz_contacts(in_path, trans=False, store_sparse=True)
    
    #if ref_chromo_limits:
    #  normalize_contacts(contacts, chromo_limits, file_bin_size, ref_chromo_limits, store_sparse=True)
    #else:
    #  ref_chromo_limits = chromo_limits
    
    normalize_contacts(contacts, chromo_limits, file_bin_size, store_sparse=True)
    
    file_counts = []
    file_deltas = []
    
    for chr_a in chromos:
    
      chromo_pair = (chr_a, chr_a)
      loops = point_dict.get(chromo_pair)
      
      if loops is None:
        continue
      
      loops = np.array(loops, float)
      nl = len(loops)
      if i == 0:
        num_loops += nl
      
      start, end = chromo_limits[chr_a]     
      mat = contacts[chromo_pair]
      
      if hasattr(mat, 'toarray'):
        mat = mat.toarray()
      
      n = len(mat)
      mat = mat.astype(float)
      msum = mat.sum()
      
      """
      medians = np.ones(n, float)
      for d in range(1, n):
        deltas = np.zeros(n-d, float)
        idx = np.array(range(n-d))
        idx = (idx, idx + d) # rows, cols

        vals = mat[idx]
        med = np.median(vals[vals > 0])
        medians[d] = med              
      """
          
      pos_1 = loops[:,0]-start
      pos_2 = loops[:,1]-start
      
      rows = (pos_1/file_bin_size).astype(int)
      cols = (pos_2/file_bin_size).astype(int)
      
      deltas = np.abs(rows-cols) # num separating bins, irrespective of validity
      counts = np.zeros(nl) # Must be same len for each Hi-C file for comparisons,  Invalid remain zero
                  
      if not msum:
        file_counts.append(counts)
        file_deltas.append(deltas)
        continue
      
      mat *= 1e9/mat.sum()
      
      dx1 = pos_1-(rows*file_bin_size)
      dy1 = pos_2-(cols*file_bin_size)
      
      dx0 = file_bin_size-dx1
      dy0 = file_bin_size-dy1
      
      # Calc weights for four bins overlapping loop square      
      sq = float(file_bin_size ** 2)
      w00 = (dx0*dy0)/sq
      w01 = (dx0*dy1)/sq
      w10 = (dx1*dy0)/sq
      w11 = (dx1*dy1)/sq
      
      valid = rows >= 0
      valid &= rows < n
      valid &= cols >= 0
      valid &= cols < n
      
      rows = rows[valid]
      cols = cols[valid]
       
      if len(rows) != len(pos_1):
        msg = 'Found {:,} out-of-bounds points, relative to contact map, in chromosome/contig {}'
        util.warn(msg.format(len(pos_1)-len(rows), chr_a))  
      
      counts[valid]  = w11[valid] * mat[(rows,cols)] + w01[valid] * mat[(rows-1,cols)] \
                       + w10[valid] * mat[(rows,cols-1)] + w00[valid] * mat[(rows-1,cols-1)]      
      if tsv_path:
        counts_dict[chromo_pair].append(counts)
      
      #counts /= medians[deltas]

      file_counts.append(counts)
      file_deltas.append(deltas)

    file_deltas = np.concatenate(file_deltas)
    file_counts = np.concatenate(file_counts)
    all_counts.append(file_counts)
    all_deltas.append(file_deltas)
  
  
  ref_counts = all_counts[0]
  ref_nz = ref_counts > 0
    
  if tsv_path:
    util.warn('Only separations larger than the Hi-C bin size are written to TSV.')  
    with open(tsv_path, 'w') as file_obj:
      write = file_obj.write
      head = ['chr','size','pos_a','pos_b']
      head += ['ncount_%d(%s)' % (i+1,x) for i,x in enumerate(labels)]
      head += ['diff_1:%d' % (i+1) for i in range(1,n_inp)]
      head += ['pval_1:%d' % (i+1) for i in range(1,n_inp)]
      write('\t'.join(head) + '\n')
      
      for chromo_pair in point_dict:
        loops = point_dict.get(chromo_pair)
      
        if loops is None:
          continue
          
        chromo = chromo_pair[0]
        counts = counts_dict[chromo_pair]
        n = len(counts)
        
        for i, region in enumerate(loops):
           start, end = sorted(region)
           size = end-start
           
           if size > file_bin_size:
             row = [chromo, '%d' % size, '%d' % start, '%d' % end]
             row_counts = [counts[j][i] for j in range(n)]
             
             for count in row_counts:
               row.append('%d' % count)
 
             for count in row_counts[1:]:
               row.append('%d' % (row_counts[0]-count))
             
             for count in row_counts[1:]:
               zstat, pval = poisson_wald(row_counts[0], count)
               if pval is None:
                 row.append('-')
               else:
                 row.append('%.5e' % pval)
           
             write('\t'.join(row) + '\n')
             
    util.info('Written {}'.format(tsv_path))
          
  from colorsys import hsv_to_rgb
  
  max_count = max([x.max() for x in all_counts])
  max_count = np.ceil(10.0*np.log10(max_count))/10.0
  
  min_count = max([x[x > 0].min() for x in all_counts ])
  min_count = np.floor(10.0*np.log10(min_count))/10.0
  
  ss_min = min([x.min() for x in all_deltas])*file_bin_size/1e3 
  ss_max = max([x.max() for x in all_deltas])*file_bin_size/1e3 
  
  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(pdf_path)
    
  colors = [hsv_to_rgb(h, 1.0, 0.5) for h in np.arange(0.0, 0.8, 1.0/n_inp)] 
  colors = ['#%02X%02X%02X' % (r*255, g*255, b*255) for r,g,b in colors]
  
  score_range = (min_count, max_count)
 
  fig = plt.figure()
  fig.set_size_inches(10.0, 8.0)
  ax1 = fig.add_axes([0.1, 0.10, 0.8, 0.8])
  
  pair_file_name = os.path.basename(paired_region_path)
  
  if len(pair_file_name) > 80:
    pair_file_name = pair_file_name[:80] + '...'

  ax1.set_title('Hi-C count distributions ({:.1f} kb bins) for {:,} paired points\nin file {}'.format(file_bin_size/1e3, num_loops, pair_file_name))
  
  ax1.set_xlabel('Normalised contact $log_{10}$(count)')
  ax1.set_ylabel('Probability density')
  ax1.set_xlim(score_range)
  n_bins = 100
  
  for i, counts in enumerate(all_counts):
    nz = (counts != 0)
    hist, edges = np.histogram(np.log10(counts[nz]), normed=True, bins=n_bins, range=score_range)
    ax1.plot(edges[:-1], hist, alpha=0.5, linewidth=2, label=labels[i], color=colors[i])
  
  ax1.legend(loc=2)
  
  if pdf:
    pdf.savefig(dpi=100)
  else:
    plt.show() 
  
  fig = plt.figure()
  fig.set_size_inches(10.0, 8.0)
  ax = fig.add_axes([0.1, 0.10, 0.8, 0.8])
  ax.set_title('Average Hi-C count ratios ({:.1f} kb bins) vs sequence separation'.format(file_bin_size/1e3))
  
  for i, counts in enumerate(all_counts[1:], 1):
    nz = (counts > 0) & ref_nz
  
    ratios = np.log10(counts[nz]/ref_counts[nz])    
    seps = all_deltas[i][nz] 
    
    data_dict = defaultdict(list) 
    for j, sep in enumerate(seps):
      data_dict[sep].append(ratios[j])

    x_vals = []
    means = []
    sems = []
    for sep in range(int(max_sep/file_bin_size)):
      if sep in data_dict:
        col_data = data_dict[sep]
        means.append(np.mean(col_data))
        sems.append(sem(col_data))
        x_vals.append(file_bin_size/1e3 * sep)
    
    means = np.array(means)
    sems = np.array(sems)
    ax.plot(x_vals, means, alpha=0.5, label='%s/%s' % (labels[i],labels[0]), linewidth=2, color=colors[i])
    ax.errorbar(x_vals, means, sems, alpha=0.5, color=colors[i], label='SEM')

  ax.set_xlabel('Seq. separation (kb)')
  ax.set_ylabel('Mean $log$(count_ratio)')
   
  ax.plot([ss_min, ss_max], [0.0, 0.0], color='#808080', alpha=0.5, linestyle='--', linewidth=2) 
  ax.legend(loc=4)
  
  if pdf:
    pdf.savefig(dpi=100)
  else:
    plt.show() 
  
  fig = plt.figure()
  fig.set_size_inches(10.0, 8.0)
  ax2 = fig.add_axes([0.1, 0.10, 0.8, 0.8])
  ax2.set_title('Average Hi-C count ratios ({:.1f} kb bins) vs $log_{{10}}$(sequence separation)'.format(file_bin_size/1e3))
  
  for i, counts in enumerate(all_counts[1:], 1):
    nz = (counts > 0) & ref_nz
  
    ratios = np.log10(counts[nz]/ref_counts[nz])    
    seps = all_deltas[i][nz] 
    
    data_dict = defaultdict(list) 
    for j, sep in enumerate(seps):
      data_dict[sep].append(ratios[j])

    x_vals = []
    means = []
    sems = []
    for sep in range(int(max_sep/file_bin_size)):
      if sep in data_dict:
        col_data = data_dict[sep]
        means.append(np.mean(col_data))
        sems.append(sem(col_data))
        x_vals.append(file_bin_size * sep)
    
    means = np.array(means)
    sems = np.array(sems)
    ax2.plot(np.log10(x_vals), means, alpha=0.5, label='%s/%s' % (labels[i],labels[0]), linewidth=2, color=colors[i])
    ax2.errorbar(np.log10(x_vals), means, sems, alpha=0.5, color=colors[i], label='SEM')

  ax2.set_xlabel('$log_{10}$(Seq. separation)')
  ax2.set_ylabel('Mean $log$(count_ratio)')

  ax2.plot([np.log10(file_bin_size), np.log10(1e3*ss_max)], [0.0, 0.0], color='#808080', alpha=0.5, linestyle='--', linewidth=2) 
  ax2.legend(loc='lower left')
  
  if pdf:
    pdf.savefig(dpi=100)
  else:
    plt.show() 
  
  for i, counts in enumerate(all_counts[1:], 1):
    fig = plt.figure()
    fig.set_size_inches(10.0, 8.0)
    
    ax = fig.add_axes([0.1, 0.10, 0.8, 0.8])
    ax.set_title('%s vs %s same bin comparison (%.1f kb)' % (labels[0], labels[i], file_bin_size/1e3))
    ax.set_xlabel('Normalised $log_{10}$(count), %s' % labels[0])
    ax.set_ylabel('Normalised $log_{10}$(count), other')
 
    cmap =  LinearSegmentedColormap.from_list(name='pcm%d' % i, colors=['#FFFFFF', colors[i], '#000000', '#FFFFFF'], N=255)
    cmap.set_bad('#FFFFFF')
    
    nz = (counts > 0) & ref_nz

    counts, xedges, yedges, img = ax.hist2d(np.log10(ref_counts[nz]), np.log10(counts[nz]), range=(score_range, score_range),
                                            label=labels[i], cmap=cmap, bins=(100,100), norm=LogNorm())
    cbar = plt.colorbar(img, ax=ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Bin count', fontsize=9)
 
    ax.plot(score_range, score_range, color='#808080', alpha=0.5)
    ax.set_xlim(score_range)
    ax.set_ylim(score_range)
    
    if pdf:
      pdf.savefig(dpi=100)
    else:
      plt.show()
  
  if pdf:
    plt.close()
    pdf.close()
    util.info('Written {}'.format(pdf_path))
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

  arg_parse.add_argument(metavar='PAIRED_POINT_FILE', nargs=1, dest="r",
                         help='Data track file in BED format specifying the paired chromosome analysis positions')

  arg_parse.add_argument(metavar='CONTACT_FILES', nargs='+', dest="i",
                         help='Input NPZ or NCC format chromatin contact file(s). Wildcards accepted')

  arg_parse.add_argument('-o', '--out-pdf', metavar='PDF_FILE', default=None, dest="o",
                         help='Output PDF format file. If not specified, a default based on the input file name(s).')

  arg_parse.add_argument('-t', '--out-tsv', metavar='TSV_FILE', default=None, dest="t",
                         help='Output TSV format text file listing regions, sizes and contact counts.')

  arg_parse.add_argument('-g', '--gfx', default=False, action='store_true', dest="g",
                         help='Display graphics on-screen using matplotlib and do not automatically save output.')

  arg_parse.add_argument('-l', '--labels', metavar='LABELS', nargs='*', dest="l",
                         help='Text labels for the input files (otherwise the input file names wil be used)')

  arg_parse.add_argument('-s', '--bin-size', default=DEFAULT_BIN_SIZE, metavar='KB_BIN_SIZE', type=int, dest="s",
                         help='When using NCC format input, the sequence region size in kilobases for calculation of contact counts. Default is %d (kb)' % DEFAULT_BIN_SIZE)

 
  args = vars(arg_parse.parse_args(argv))

  paired_region_path = args['r'][0]
  contact_paths = args['i']
  pdf_path = args['o']
  tsv_path = args['t']
  bin_size = args['s']
  labels = args['l'] or None
  screen_gfx = args['g']
  
  #num_bootstrap = args['nb']
  #num_null = args['nn']
  
  for file_path in contact_paths:
    io.check_invalid_file(file_path, critical=True)
   
  if pdf_path and screen_gfx:
    util.warn('Output PDF file will not be written in screen graphics (-g) mode')
    pdf_path = None
     
  contact_points(paired_region_path, contact_paths, pdf_path, bin_size,
                 labels, screen_gfx, tsv_path)
  

if __name__ == "__main__":
  
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()


"""
./nuc_tools contact_pair_points ES_loops_mm10.bed /data/hi-c/pop/SLX-7672_E14_100k.npz /data/hi-c/pop/SLX-7676_Mbd3KO_100k.npz -l E14 Mbd3KO -g
"""

