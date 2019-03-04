import sys,  os, math
import numpy as np
from matplotlib import pyplot as plt
from colorsys import hsv_to_rgb

PROG_NAME = 'contact_probability'
VERSION = '1.0.0'
DESCRIPTION = 'Chromatin contact (NCC or NPZ format) probability vs sequence separation and region analysis'
DEFAULT_BIN_SIZE = 25
REGION_TILES = ('Within','Overlap','Outside')

def contact_probability(contact_paths, pdf_path=None, region_path=None, bin_size=DEFAULT_BIN_SIZE,
                        labels=None, region_label=None, max_sep=1e8):

  from nuc_tools import util, io
  from formats import ncc, npz, bed
  
  if not pdf_path:
    pdf_path = os.path.splitext(contact_paths[0])[0] + '_contact_prob.pdf' 
    
  pdf_path = io.check_file_ext(pdf_path, '.pdf')
  
  if labels:
    for i, label in enumerate(labels):
      labels[i] = label.replace('_',' ')
      
    while len(labels) < len(contact_paths):
      labels.append(os.path.basename(contact_paths[len(labels)]))
      
  else:
    labels = [os.path.basename(x) for x in contact_paths]
  
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
  
  colors = [hsv_to_rgb(h, 1.0, 0.8) for h in np.arange(0.0, 0.8, 1.0/len(contact_paths))] 
  colors = ['#%02X%02X%02X' % (r*255, g*255, b*255) for r,g,b in colors]
      
  bin_size *= 1e3
  
  if region_dict:
    f, axarr = plt.subplots(2,2)
    ax0 = axarr[0,0]
    ax1 = axarr[0,1]
    ax2 = axarr[1,0]
    ax3 = axarr[1,1]
    axes = (ax0, ax1, ax2, ax3)
    
  else:
    f, ax0 = plt.subplots()
    axes = [ax0]
  x_limit = max_sep
  num_bins = (x_limit-bin_size)/bin_size
  bins = np.linspace(2*bin_size, x_limit, num_bins)

  y_mins = []
  y_maxs = []
   
  for i, in_path in enumerate(contact_paths):
    util.info('Processing %s (%s)' % (in_path, labels[i]))
    seq_seps = []
    weights = []
    
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
        
        for d in range(0, n):
          m = n-d
          rows = np.array(range(m))
          cols = rows + d
          idx = (rows, cols)
 
          frac = n/float(m)
          ss = np.full(m, d * file_bin_size)
          wt = matrix[idx] * frac # - weights come from fraction of chromo and sum of counts
 
          nz = wt.nonzero()
          ss = ss[nz]
          wt = wt[nz]
          pa = chromo_limits[chr_a][0] + rows * file_bin_size
          pb = chromo_limits[chr_a][0] + cols * file_bin_size
          
          if len(wt):
            prob.append(wt)
            seps.append(ss)
            seq_pos_a.append(pa[nz])
            seq_pos_b.append(pb[nz])
        
        seps = np.concatenate(seps)
        prob = np.concatenate(prob)
        seq_pos_a = np.concatenate(seq_pos_a)
        seq_pos_b = np.concatenate(seq_pos_b)
       
      else:
        contact_array = contacts[chromo]
        seq_pos_a = contact_array[:,0]
        seq_pos_b = contact_array[:,1]
        
        seps = abs(seq_pos_a-seq_pos_b)
        indices = seps.nonzero()
        seps = seps[indices]

        p_start, p_end = chromo_limits[chromo]
        size = float(p_end-p_start+1)

        prob = (size/(size-seps)) # From fraction of chromosome that could give rise to each separation
        seps = seps
       
      if len(prob):
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
    
    seq_seps = np.concatenate(seq_seps)
    weights = np.concatenate(weights)
    
    util.info('  .. found {:,} values'.format(len(seq_seps)))

    hist, edges = np.histogram(seq_seps, bins=bins, weights=weights, density=True)

    idx = hist.nonzero()

    hist = hist[idx]
    edges = edges[idx]

    x_data = np.log10(edges)
    y_data = np.log10(hist)

    y_mins.append(y_data.min())
    y_maxs.append(y_data.max())
    
    ax0.plot(x_data, y_data, label=labels[i], color=colors[i], linewidth=1, alpha=0.5)
    
    if region_dict:
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
  
  ax0.set_title('Contact sequence separations')
  
  for ax in axes:
    ax.set_alpha(0.5)

    ax.set_xlabel('Sequence separation (bp)')
    ax.set_ylabel('Contact probability (%d kb bins)' % (bin_size/1e3))
    ax.xaxis.set_ticks(x_range)
    ax.set_xticklabels(['$10^{%.1f}$' % x for x in x_range], fontsize=12)
    ax.set_xlim((x_min, np.log10(x_limit)))
 
    ax.yaxis.set_ticks(y_range)
    ax.set_yticklabels(['$10^{%.1f}$' % x for x in y_range], fontsize=12)
    ax.set_ylim((y_min, y_max))

    ax.plot([5.5, 7.0], [-6.0, -7.50], color='#808080', alpha=0.5, linestyle='--')
    ax.plot([5.5, 7.0], [-6.5, -8.75], color='#808080', alpha=0.5, linestyle='--')
    ax.text(7.0, -7.50, '$\lambda=1.0$', color='#808080', verticalalignment='center', alpha=0.5, fontsize=14)
    ax.text(7.0, -8.75, '$\lambda=1.5$', color='#808080', verticalalignment='center', alpha=0.5, fontsize=14)

    ax.legend()

  #plt.savefig(svg_path)
  plt.show()
  
  
  

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
  
  contact_probability(contact_paths, pdf_path, region_path, bin_size, labels, region_label)
  
  # Add -r, --regions
  # - Plot contact probability separately for
  #   - Both ends in, both ends out, one end in
  #   - For each dataset
  #   - For each for the three classes, multiplexing on dataset    
    

if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()
