import sys, math, os
import numpy as np
from random import randint
from matplotlib import pyplot as plt

PROG_NAME = 'contact_enrichment'
VERSION = '1.0.0'
DESCRIPTION = 'Within vs between region chromatin contact (NCC or NPZ format) analysis'
DEFAULT_BIN_SIZE = 100
DEFAULT_NULL_SAMPLES = 10
DEFAULT_BOOTSTRAP_SAMPLES = 1000

# Are TADs/compartments stronger or weaker?

# - Plot contact probability within and between regions
#   - Inter/intra Enrichment vs seq separation
#     + Within/between count ratio compared to permutation null 
#     + Bootstrap errors
#   - Plot distributions of partitioning inter/intra for each input
#     + Each point/bin has an intra/inter ratio 
#     + Histogram of ratios at different seq sep thresholds
#     + Scatter of ratios for two samples
# ? Do seq sep analysis too?

def get_obs_vs_exp(obs, clip=10):

  from contact_map import get_cis_expectation

  obs -= np.diag(np.diag(obs))
  expt = get_cis_expectation(obs)

  prod = expt * obs

  nz = prod != 0.0
 
  log_ratio = obs.copy()
  log_ratio[nz] /= expt[nz]
  log_ratio[nz] = np.log(log_ratio[nz])
  log_ratio = np.clip(log_ratio, -clip, clip)
  
  
  return log_ratio
  
def contact_enrichment(region_path, contact_paths, pdf_path, bin_size=DEFAULT_BIN_SIZE, labels=None,
                       num_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES, num_null=DEFAULT_NULL_SAMPLES):

  from nuc_tools import util, io
  from formats import bed, ncc, npz  
  from contact_compare import normalize_contacts
  
  bin_size *= 1000
  
  if labels:
    while len(labels) < len(contact_paths):
      labels.append(os.path.basename(contact_paths[len(labels)]))
  else:
    labels = [os.path.basename(x) for x in contact_paths]
  
  region_dict, value_dict, label_dict = bed.load_bed_data_track(region_path)
  
  for chr_a in region_dict:
    ends = region_dict[chr_a][:,0]
    idx = ends.argsort()
    region_dict[chr_a] = region_dict[chr_a][idx,:]

    
  prev_ratios = None
  
  for i, in_path in enumerate(contact_paths):
    label = labels[i]
    
    if in_path.lower().endswith('.ncc') or in_path.lower().endswith('.ncc.gz'):
       file_bin_size = None
       chromosomes, chromo_limits, contacts = ncc.load_file(in_path, trans=False)
    
    else:
       file_bin_size, chromo_limits, contacts = npz.load_npz_contacts(in_path, trans=False)
       normalize_contacts(contacts, chromo_limits, file_bin_size, store_sparse=False)

    all_ratios = []
    null_ratios = []
    ww = []
    
    chromos = util.sort_chromosomes([x[0] for x in contacts])
    
    for chr_a in chromos:
      chromo_pair = (chr_a, chr_a)
      start, end = chromo_limits[chr_a]
      
      if file_bin_size:
        start = 0
  
      width = end-start
      m = int(math.ceil(width/bin_size))
      
      if file_bin_size:
        mat = get_obs_vs_exp(contacts[chromo_pair])
        rows, cols = mat.nonzero()
        seq_pos_a = start + rows * file_bin_size + file_bin_size/2
        seq_pos_b = start + cols * file_bin_size + file_bin_size/2
        counts = mat[(rows, cols)]
        
      else:
        contacts = np.array(contacts[chromo_pair]).T
        seq_pos_a = contacts[0]
        seq_pos_b = contacts[1]
        counts = contacts[2]
      
      regions = region_dict[chr_a]
      
      if not len(regions):
        continue
      
      offset = 0
      for j in range(num_null+1):
        if j % 1 == 0:
          util.info('%s %s round %d' % (label, chr_a, j), line_return=True)
        
        if j:
          deltas = regions[:,1] - regions[:,0]
          rstarts = (regions[:,0] - start + offset) % width
          rends = np.clip(rstarts + deltas, 0, width)
          rstarts += start
          rends += start
          order = rstarts.argsort()
          rstarts = rstarts[order]
          rends = rends[order]
        else:
          rstarts = regions[:,0]
          rends = regions[:,1]                            
        
        rbig = np.argmax(rends-rstarts)
         
        rstarts2 = np.append(rstarts, [end])

        e1 = np.searchsorted(rends, seq_pos_a) # Region indices for each contact
        e2 = np.searchsorted(rends, seq_pos_b)
       
        # Are seq pos at or above the region starts corresponding to the region ends that they are immediately less than
        above_start1 = seq_pos_a >= rstarts2[e1]
        above_start2 = seq_pos_b >= rstarts2[e2]
 
        #close = np.abs(seq_pos_b-seq_pos_a) < 5e6
        #far = np.abs(seq_pos_b-seq_pos_a) > 0.5e6
        valid = above_start1 & above_start2 #& far #& close # Elements where both pos are in any region
        
        same_region = e1 == e2 # Elements where closest region end is same
        
        # Select only where both positions are in a region
        seq_pos_a2 = seq_pos_a[valid]
        seq_pos_b2 = seq_pos_b[valid]
        
        counts2 = counts[valid]
        same_region = same_region[valid]
        not_same_region = ~same_region
        
        bins_a = (seq_pos_a2-start)/bin_size
        bins_b = (seq_pos_b2-start)/bin_size

        """
        e1 = e1[valid]
        e2 = e2[valid]
        
        
        idx_1 = e1 == rbig
        idx_2 = e2 == rbig
        
        idx_w = (idx_1 & idx_2).nonzero()[0]
        idx_b = (np.logical_xor(idx_1, idx_2)).nonzero()[0]
        idx_0 = idx_1 & idx_2
        idx_0 = (~idx_0).nonzero()[0]

        w = counts2[idx_w].sum()
        b = counts2[idx_b].sum()
        
        print 
        print w, b, '%.2f' % (w/float(w+b)), rstarts[rbig], rends[rbig]-rstarts[rbig]
        
        
        fig, ax = plt.subplots()
        ax.scatter(seq_pos_a2[idx_w], seq_pos_b2[idx_w], s=1, alpha=0.01, color='#FF0000')
        #ax.scatter(seq_pos_b2[idx_b], seq_pos_a2[idx_b], s=1, alpha=0.01, color='#0000FF')
        ax.scatter(seq_pos_b2[idx_0], seq_pos_a2[idx_0], s=1, alpha=0.01, color='#00B000')
        #ax1.plot(between, color='#0000FF')
        #ax2.plot((1.0+within)/(1.0+between), color='#008000')
        plt.show()
        """
        
        idx = same_region.nonzero()
        weights = counts2[idx]
        within = np.histogram(bins_a[idx], weights=weights, bins=m, range=(0,m-1))[0].astype(float)
        within += np.histogram(bins_b[idx], weights=weights, bins=m, range=(0,m-1))[0]

        idx = not_same_region.nonzero()
        weights = counts2[idx]
        between = np.histogram(bins_a[idx], weights=weights, bins=m, range=(0,m-1))[0].astype(float)
        between += np.histogram(bins_b[idx], weights=weights, bins=m, range=(0,m-1))[0]

        #ax1.hist(np.log10(1.0+counts[same_region]), bins=50, color='#FF0000')
        #ax2.hist(np.log10(1.0+counts[not_same_region]), bins=50, color='#0000FF')
        #ax.scatter(bins_a, between, s=1, alpha=0.01, color='#FF0000')
        #ax.scatter(bins_a, within, s=1, alpha=0.01, color='#0000FF')

        totals = within + between
        nz = (within * between).nonzero()
        within = within[nz]
        totals = totals[nz]
        within /= totals
        """
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(within, color='#FF0000')
        ax1.plot(between, color='#0000FF')
        ax2.scatter(within, between, color='#008000', alpha=0.25)
        plt.show()
        """
        ww.append(within)

        if j == 0:
          all_ratios.append(within)
        else:
          null_ratios.append(within)
        
        offset = (offset + 8000000) % width
        
        #if j == 0:
        #  offset += 8000000
        #else:  
        #  offset = randint(bin_size, width-1)      
    
    """
    fig, (ax1, ax2) = plt.subplots(2, 1)     
    ax1.plot(ww[0], color='#FF0000', label='Within Obs')
    ax1.plot(ww[1], color='#0000FF', label='Within Exp')
     
    plt.legend()
    plt.show()
    """
    
    all_ratios = np.concatenate(all_ratios, axis=0)
    null_ratios = np.concatenate(null_ratios, axis=0)
             
    fig, ax = plt.subplots()
    ax.set_title(label)
    
    obs, edges = np.histogram(all_ratios, normed=True, range=(-0.5, 0.5), bins=192)
    obs /= float(obs.sum())

    expt, edges = np.histogram(null_ratios, normed=True, range=(-0.5, 0.5), bins=192)
    expt /= float(expt.sum())
    
    nz = (obs * expt).nonzero()
    obs = obs[nz]
    expt = expt[nz]
    edges = edges[nz]
    
    #ax.plot(edges[:-1], obs, alpha=0.3, color='#0080FF', label='Data') 
    #ax.plot(edges[:-1], expt, alpha=0.3, color='#FF4000', label='Null') 
    
    h = obs * np.log(obs/expt)
    
    ax.plot(edges, h, alpha=0.3, color='#0080FF', label='Entropy')
    
    ax.legend()
    
    print "H: %.4f" % h.sum()
    
    plt.show()
    
      
def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)

  arg_parse.add_argument(metavar='REGION_FILE', nargs=1, dest="r",
                         help='Data track file in BED format specifying chromosome analysis regions')

  arg_parse.add_argument(metavar='CONTACT_FILES', nargs='+', dest="i",
                         help='Input NPZ or NCC format chromatin contact file(s). Wildcards accepted')

  arg_parse.add_argument('-o', metavar='PDF_FILE', default=None, 
                         help='Output PDF format file. If not specified, a default based on the input file name(s).')

  arg_parse.add_argument('-g', default=False, action='store_true',
                         help='Display graphics on-screen using matplotlib and do not automatically save output.')

  arg_parse.add_argument('-l', metavar='LABELS', nargs='*',
                         help='Text labels for the input files (otherwise the input file names wil be used)')

  arg_parse.add_argument('-s', default=DEFAULT_BIN_SIZE, metavar='KB_BIN_SIZE', type=int,
                         help='When using NCC format input, the sequence region size in kilobases for calculation of contact enrichments. Default is %d (kb)' % DEFAULT_BIN_SIZE)

  arg_parse.add_argument('-nb', default=DEFAULT_BOOTSTRAP_SAMPLES, metavar='NUM_BOOTSTRAP_SAMPLES', type=int,
                         help='Number of resamplings to perform for bootstrapped error estimates. Default is %d' % DEFAULT_BOOTSTRAP_SAMPLES)

  arg_parse.add_argument('-nn', default=DEFAULT_NULL_SAMPLES, metavar='NUM_NULL_SAMPLES', type=int,
                         help='Number of times regions are shifted randmolnly to create the background/null expectation. Default is %d' % DEFAULT_NULL_SAMPLES)


 
  args = vars(arg_parse.parse_args(argv))

  region_path = args['r'][0]
  contact_paths = args['i']
  pdf_path = args['o']
  bin_size = args['s']
  labels = args['l'] or None
  num_bootstrap = args['nb']
  num_null = args['nn']
  
  for file_path in contact_paths:
    invalid_msg = io.check_invalid_file(file_path)
    if invalid_msg:
      util.critical(invalid_msg)
    
  contact_enrichment(region_path, contact_paths, pdf_path, bin_size, labels, num_bootstrap, num_null)
  

if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()
