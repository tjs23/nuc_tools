import sys, math, os
import numpy as np
from collections import defaultdict
from random import randint
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROG_NAME = 'contact_points'
VERSION = '1.0.0'
DESCRIPTION = 'Paired-point chromatin contact (NPZ format) analysis'
DEFAULT_BIN_SIZE = 100
  
def contact_points(paired_region_path, contact_paths, pdf_path, bin_size=DEFAULT_BIN_SIZE, labels=None,
                   screen_gfx=False):

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
  
  n_inp = len(contact_paths)
  all_counts = []
  all_deltas = []
  
  for i, in_path in enumerate(contact_paths):
    
    file_bin_size, chromo_limits, contacts = npz.load_npz_contacts(in_path, trans=False, store_sparse=True)
    normalize_contacts(contacts, chromo_limits, file_bin_size, store_sparse=True)
    
    chromos = util.sort_chromosomes([x[0] for x in contacts])
    file_counts = []
    file_deltas = []
    
    for chr_a in chromos:
      chromo_pair = (chr_a, chr_a)
      loops = point_dict.get(chromo_pair)
      
      if loops is None:
        continue
      
      loops = np.array(loops, float)
      
      start, end = chromo_limits[chr_a]     
      mat = contacts[chromo_pair]
      
      if hasattr(mat, 'toarray'):
        mat = mat.toarray()
      
      mat = mat.astype(float)
      msum = mat.sum()
      
      if not msum:
        continue
      
      mat *= 1e7/msum
      
      n = len(mat)
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
        
      pos_1 = loops[:,0]
      pos_2 = loops[:,1]
      
      rows = ((pos_1-start)/file_bin_size).astype(int)
      cols = ((pos_2-start)/file_bin_size).astype(int)
      
      counts = mat[(rows,cols)]
      
      deltas = np.abs(rows-cols) # num separating bins
      
      #counts /= medians[deltas]

      file_counts.append(counts)
      file_deltas.append(deltas)
    
    file_deltas = np.concatenate(file_deltas)
    file_counts = np.concatenate(file_counts)
    all_counts.append(file_counts)
    all_deltas.append(file_deltas)
 
  from colorsys import hsv_to_rgb
  
  max_count = max([x.max() for x in all_counts])
  max_count = np.ceil(np.log10(max_count))
  
  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(pdf_path)
    
  colors = [hsv_to_rgb(h, 1.0, 0.5) for h in np.arange(0.0, 0.8, 1.0/n_inp)] 
  colors = ['#%02X%02X%02X' % (r*255, g*255, b*255) for r,g,b in colors]
  
  score_range = (0.0, max_count)
      
  fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
  
  ax1.set_title('Hi-C counts (%d kb bins) at %s points' % (file_bin_size/1e3, os.path.basename(paired_region_path)))
  
  ax1.set_xlabel('Normalised contact $log_{10}$(count)')
  ax1.set_ylabel('Probability density')
  #ax1.set_xlim(score_range)
  n_bins = int(max_count) * 5
  
  for i, counts in enumerate(all_counts):
    nz = (counts > 0)
    hist, edges = np.histogram(np.log10(counts[nz]), normed=True, bins=n_bins) #  , range=score_range)
    ax1.plot(edges[:-1], hist, alpha=0.5, linewidth=2, label=labels[i], color=colors[i])
  
  ax1.legend(loc=2)
  
  ref_counts = all_counts[0]
  
  ax2.set_xlabel('Normalised $log_{10}$(count), %s' % labels[0])
  ax2.set_ylabel('Normalised $log_{10}$(count), other')
  ref_nz = ref_counts > 0
  
  for i, counts in enumerate(all_counts[1:], 1):
    nz = (counts > 0) & ref_nz
    ax2.scatter(np.log10(ref_counts[nz]), np.log10(counts[nz]), alpha=0.2, s=2, label=labels[i], color=colors[i])
  
  ax2.plot(score_range, score_range, color='#808080', alpha=0.5) 
  ax2.set_xlim(score_range)
  ax2.set_ylim(score_range)
  ax2.legend(loc=4)
  
  max_sep = 2e6
  
  for i, counts in enumerate(all_counts[1:], 1):
    nz = (counts > 0) & ref_nz
  
    ratios = np.log10(counts[nz]/ref_counts[nz])
    seps = all_deltas[i][nz] 
    
    data_dict = defaultdict(list) 
    for j, sep in enumerate(seps):
      data_dict[sep].append(ratios[j])
    
    lquart = []
    uquart = []
    medians = [] 
    x_vals = []
    means = []
    sems = []
    for sep in range(int(max_sep/file_bin_size)):
      if sep in data_dict:
        col_data = data_dict[sep]
        q25, q50, q75 = np.percentile(col_data, [25.0, 50.0, 75.0])
        medians.append(q50)
        lquart.append(q25)
        uquart.append(q75)
        means.append(np.mean(col_data))
        sems.append(stats.sem(col_data))
        x_vals.append(file_bin_size/1e3 * sep)
    
    medians = np.array(medians)
    lquart = np.array(lquart)
    uquart = np.array(uquart)
    means = np.array(means)
    sems = np.array(sems)
    
    yerr = [medians-lquart, uquart-medians]
    
    ax3.plot(x_vals, means, alpha=0.3, label='%s/%s' % (labels[i],labels[0]), color=colors[i])
    ax3.errorbar(x_vals, means, sems, alpha=0.3, color=colors[i])

  ax3.set_xlabel('Seq. separation (kb)')
  ax3.set_ylabel('Median log(count_ratio)')
  
  ax3.plot([0, max_sep/1e3], [0.0, 0.0], color='#808080', alpha=0.5) 
  ax3.legend(loc=4)
  
  if pdf:
    pdf.savefig(dpi=100)
    plt.close()
    pdf.close()
    util.info('Written {}'.format(pdf_path))
  else:
    plt.show() 
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
                 labels, screen_gfx)
  

if __name__ == "__main__":
  
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()


"""
python tools/contact_pair_points.py ES_loops_mm10.bed /data/hi-c/pop/SLX-7672_E14_25k.npz /data/hi-c/pop/SLX-7676_Mbd3KO_25k.npz /data/hi-c/pop/SLX-7677_CHD4KD_25k.npz -l E14 Mbd3KO Chd4KD -g
"""

