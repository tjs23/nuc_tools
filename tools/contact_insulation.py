import sys, math, os
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROG_NAME = 'contact_insulation'
VERSION = '1.0.0'
DESCRIPTION = 'Region boundary insulation chromatin contact (NCC or NPZ format) analysis'
DEFAULT_BIN_SIZE = 100
#DEFAULT_NULL_SAMPLES = 10
#DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_KB_MAX_SEQ_SEP = 2000
MIN_BOUNDARY_SEP = 100 # Ignore close boundaries
FILE_TAG = 'hi-c_insulation'
  
def contact_insulation(region_path, contact_paths, pdf_path, bin_size=DEFAULT_BIN_SIZE, labels=None,
                       max_sep=DEFAULT_KB_MAX_SEQ_SEP, use_starts=True, use_ends=True, screen_gfx=False,
                       write_bed=False):

  from nuc_tools import util, io
  from formats import bed, ncc, npz  
  from contact_compare import normalize_contacts
  
  max_sep *= 1000
  bin_size *= 1000
  
  if not pdf_path:
    pdf_path = '%s_%s.pdf' % (os.path.splitext(contact_paths[0])[0], FILE_TAG)
    
  pdf_path = io.check_file_ext(pdf_path, '.pdf')

  if not use_starts or use_ends:
    use_starts = use_ends = True
  
  if labels:
    for i, label in enumerate(labels):
      labels[i] = label.replace('_',' ')
      
    while len(labels) < len(contact_paths):
      labels.append(os.path.basename(contact_paths[len(labels)]))
  else:
    labels = [os.path.basename(x) for x in contact_paths]
  
  
  region_dict, value_dict, label_dict = bed.load_bed_data_track(region_path)
  
  for chr_a in region_dict:
    ends = region_dict[chr_a][:,0]
    idx = ends.argsort()
    region_dict[chr_a] = region_dict[chr_a][idx,:]
  
  n_inp = len(contact_paths)
  all_ratios = []
  
  for i, in_path in enumerate(contact_paths):
    
    if io.is_ncc(in_path):
       is_ncc = True
       chromosomes, chromo_limits, contacts = ncc.load_file(in_path, trans=False)
       
       # TBD bin_contacts
        
    else:
       is_ncc = False
       bin_size, chromo_limits, contacts = npz.load_npz_contacts(in_path, trans=False)
       normalize_contacts(contacts, chromo_limits, bin_size, store_sparse=False)
    
    chromos = util.sort_chromosomes([x[0] for x in contacts])
    ratios = []
    out_region_dict = {}
    out_value_dict = {}
    
    for chr_a in chromos:
      chromo_pair = (chr_a, chr_a)
      regions = region_dict[chr_a]
      chromo_ratios = []
      
      if not len(regions):
        continue
              
      if not is_ncc:
        start = 0
      else:
        start, end = chromo_limits[chr_a]     
             
      mat = contacts[chromo_pair].astype(float)
      msum = mat.sum()
      
      if not msum:
        continue
      
      mat /= 1e7 * msum
      n = mat.shape[0]
      
      dbin = int(max_sep/bin_size)
      
      if use_starts and use_ends:
        boundaries = regions.ravel()

      elif use_starts:
        boundaries = regions[:,0]
        
      else: # ends only
        boundaries = regions[:,1]
      
      prev_pos = -10 * MIN_BOUNDARY_SEP
      valid = []
      
      for i, pos in enumerate(boundaries):
        if (pos - prev_pos) < MIN_BOUNDARY_SEP:
          continue
      
        bin_0 = (pos-start)/bin_size
        bin_a = max(0, bin_0-dbin)
        bin_b = min(n-1, bin_0+dbin)
        
        sub_mat = mat[bin_a:bin_b+1,bin_a:bin_b+1]
        mid = bin_0-bin_a # Index of boundary bin in sub-array
        idx = np.indices(sub_mat.shape)
        rows, cols = idx
        
        prev = sub_mat[(rows <= mid) & (cols <= mid)]
        next = sub_mat[(rows >= mid) & (cols >= mid)]
        
        idx_u = (rows <= mid) & (cols >= mid) & (cols-rows <= dbin)
        idx_l = (cols <= mid) & (rows >= mid) & (rows-cols <= dbin)
        inter = sub_mat[idx_u | idx_l]

        if len(inter) and len(prev) and len(next):
          im = inter.mean()
          
          if im:
            ratio = max(prev.mean(), next.mean())/im
          else:
            ratio = 0.0
            
          chromo_ratios.append(ratio)
          valid.append(i)
        
        prev_pos = pos
      
      ratios += chromo_ratios
      
      if write_bed:
        pos = boundaries[valid]
        out_region_dict[chr_a] = np.array([pos,pos+1]).T
        out_value_dict[chr_a] = np.array(chromo_ratios)
        
    ratios = np.array(ratios)
    all_ratios.append(np.log10(1.0 + ratios))
    
    if write_bed:
      bed_path = '%s_%s_%s.bed' % (os.path.splitext(in_path)[0], os.path.splitext(os.path.basename(region_path))[0], FILE_TAG)
      bed.save_bed_data_track(bed_path, out_region_dict, out_value_dict, as_float=True)
      util.info('Written {}'.format(bed_path))
    
  from colorsys import hsv_to_rgb
  
  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(pdf_path)
    
  colors = [hsv_to_rgb(h, 1.0, 0.8) for h in np.arange(0.0, 0.8, 1.0/n_inp)] 
  colors = ['#%02X%02X%02X' % (r*255, g*255, b*255) for r,g,b in colors]
  
  score_range = (0.45, 1.2)
  
  fig = plt.figure()
  fig.set_size_inches(10.0, 5.0)
  
  ax1 = fig.add_axes([0.1, 0.15, 0.35, 0.75])
  ax2 = fig.add_axes([0.55, 0.15, 0.35, 0.75])
  
  ax1.set_title('Insulation distributions')
  ax1.set_xlabel('Insulation score')
  ax1.set_ylabel('Probability density')
  ax1.set_xlim(score_range)
  
  for i, ratios in enumerate(all_ratios):
    hist, edges = np.histogram(ratios, normed=True, bins=100)
    ax1.plot(edges[:-1], hist, alpha=0.5, linewidth=2, label=labels[i], color=colors[i])
  
  ax1.legend()
  
  ref_ratios = all_ratios[0]
  ax2.set_title('Boundary comparison')
  ax2.set_xlabel('Insulation score, %s' % labels[0])
  ax2.set_ylabel('Insulation score, other')
  
  for i, ratios in enumerate(all_ratios[1:], 1):
    ax2.scatter(ref_ratios, ratios, alpha=0.3, s=3, label=labels[i], color=colors[i])
  
  ax2.plot(score_range, score_range, color='#808080', alpha=0.5) 
  ax2.set_xlim(score_range)
  ax2.set_ylim(score_range)
  ax2.legend()

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

  arg_parse.add_argument(metavar='REGION_FILE', nargs=1, dest="r",
                         help='Data track file in BED format specifying chromosome analysis regions or boundary positions')

  arg_parse.add_argument(metavar='CONTACT_FILES', nargs='+', dest="i",
                         help='Input NPZ or NCC format chromatin contact file(s). Wildcards accepted')

  arg_parse.add_argument('-o', '--out-pdf', metavar='PDF_FILE', default=None, dest="o",
                         help='Output PDF format file. If not specified, a default based on the input file name(s).')

  arg_parse.add_argument('-b', '--write-bed', action='store_true', dest="b",
                         help='Write out insulation scores for each input contat file as a BED format file.')

  arg_parse.add_argument('-g', '--gfx', default=False, action='store_true', dest="g",
                         help='Display graphics on-screen using matplotlib and do not automatically save output.')

  arg_parse.add_argument('-l', '--labels', metavar='LABELS', nargs='*', dest="l",
                         help='Text labels for the input files (otherwise the input file names wil be used)')

  arg_parse.add_argument('-s', '--bin-size', default=DEFAULT_BIN_SIZE, metavar='KB_BIN_SIZE', type=int, dest="s",
                         help='When using NCC format input, the sequence region size in kilobases for calculation of contact enrichments. Default is %d (kb)' % DEFAULT_BIN_SIZE)

  arg_parse.add_argument('-m', '--max-seq-sep', default=DEFAULT_KB_MAX_SEQ_SEP, metavar='KB_MAX_SEQ_SEP', type=int, dest="m",
                         help='The analysis width: the maximum sequence separation to consider. Default is %d (kb)' % DEFAULT_KB_MAX_SEQ_SEP)

  arg_parse.add_argument('-start', default=False, action='store_true',
                         help='Use the boundaries at the start of each input region. Otherwise the default is to use all starts/ends unless immediately adjascent.')

  arg_parse.add_argument('-end', default=False, action='store_true',
                         help='Use the boundaries at the end of each input region. Otherwise the default is to use all starts/ends unless immediately adjascent.')

  """
  arg_parse.add_argument('-nb', default=DEFAULT_BOOTSTRAP_SAMPLES, metavar='NUM_BOOTSTRAP_SAMPLES', type=int,
                         help='Number of resamplings to perform for bootstrapped error estimates. Default is %d' % DEFAULT_BOOTSTRAP_SAMPLES)

  arg_parse.add_argument('-nn', default=DEFAULT_NULL_SAMPLES, metavar='NUM_NULL_SAMPLES', type=int,
                         help='Number of times regions are shifted randmolnly to create the background/null expectation. Default is %d' % DEFAULT_NULL_SAMPLES)
  """

 
  args = vars(arg_parse.parse_args(argv))

  region_path = args['r'][0]
  contact_paths = args['i']
  pdf_path = args['o']
  bin_size = args['s']
  labels = args['l'] or None
  use_starts = args['start']
  use_ends = args['start']
  max_sep = args['m']
  screen_gfx = args['g']
  write_bed = args['b']
  
  #num_bootstrap = args['nb']
  #num_null = args['nn']
  
  for file_path in contact_paths:
    io.check_invalid_file(file_path, critical=True)
   
  if pdf_path and screen_gfx:
    util.warn('Output PDF file will not be written in screen graphics (-g) mode')
    pdf_path = None
     
  contact_insulation(region_path, contact_paths, pdf_path, bin_size,
                     labels, max_sep, use_starts, use_ends, screen_gfx, write_bed)
  

if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()

