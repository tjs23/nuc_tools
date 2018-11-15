
import numpy as np
import os, sys

from math import ceil
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PROG_NAME = 'contact_map'
VERSION = '1.0.0'
DESCRIPTION = 'Chromatin contact (NCC format) Hi-C contact map display module'
SVG_WIDTH = 1000

OUT_FORMATS = {'PDF':'.pdf',
               'SVG':'.svg',
               'PNG':'.png',
               'JPEG':'.jpg'}
               
DEFAULT_FORMAT = 'PDF'

def _downsample_matrix(in_array, new_shape):
    
    p, q = in_array.shape
    n, m = new_shape
    
    pad_a = n * int(1+p//n) - p
    pad_b = m * int(1+q//m) - q 
    
    in_array = np.pad(in_array, [(0,pad_a), (0,pad_b)], 'constant')
    
    p, q = in_array.shape
        
    shape = (n, p // n,
             m, q // m)
    
    return in_array.reshape(shape).sum(-1).sum(1)


def contact_map(in_path, out_path, out_format=None, bin_size=None, screen_gfx=False, white_bg=False, font=None, font_size=12, line_width=0.2, min_contig_size=None):
  
  from nuc_tools import io, util
  from formats import ncc, npz
  
  if out_format:
    if out_path:
      file_root, file_ext = os.path.splitext(out_path)
      file_ext = file_ext.lower()
      
      if file_ext != OUT_FORMATS[out_format]:
        out_path = out_path + OUT_FORMATS[out_format]
      
    else:
      file_root, file_ext = os.path.splitext(in_path)
      out_path = file_root + OUT_FORMATS[out_format]
  
  elif out_path:
    file_root, file_ext = os.path.splitext(out_path)
    file_ext = file_ext.lower()
    
    for format_name, format_ext in OUT_FORMATS.items():
      if file_ext == format_ext:
        out_format = format_name
        break
    
    else:
      out_path = file_root + OUT_FORMATS[DEFAULT_FORMAT]
      out_format =  DEFAULT_FORMAT
   
  else:
    file_root, file_ext = os.path.splitext(in_path)
    out_path = file_root + OUT_FORMATS[DEFAULT_FORMAT]   
    out_format =  DEFAULT_FORMAT
  
  if screen_gfx:
    util.info('Displaying contact map for {}'.format(in_path))
  else:
    util.info('Making contact map for {} in {} format'.format(in_path, out_format))
  
  if in_path.lower().endswith('.ncc'):
    file_bin_size = None
    chromosomes, chromo_limits, contacts = ncc.load_ncc(in_path)
    
  else:
    file_bin_size, chromo_limits, contacts = npz.load_npz_contacts(in_path)

  if not chromo_limits:
    util.fatal('No chromosome contact data read')

  if min_contig_size:
    min_contig_size = int(min_contig_size * 1e6)
  else:
    largest = max([e-s for s, e in chromo_limits.values()])
    min_contig_size = int(0.05*largest) 
    util.info('Min. contig size not specified, using 5% of largest: {:,} bp'.format(min_contig_size))
  
  if bin_size:
    bin_size = int(bin_size * 1e6)
     
  else:
    tot_size = 0
    
    for chromo in chromo_limits:
      s, e = chromo_limits[chromo]
      size = e-s
      
      if size >= min_contig_size:
        tot_size += size 
    
    bin_size = int(tot_size/1000)
    util.info('Bin size not specified, using approx. 1000 x 1000 bin equivalent: {:,} bp'.format(bin_size))
  
      
  # Get sorted chromosomes, ignore small contigs as appropriate
  chromos = []
  skipped = []
  for chromo in chromo_limits:
    s, e = chromo_limits[chromo]

    if (e-s) < min_contig_size:
      skipped.append(chromo)
      continue

    if chromo.lower().startswith('chr'):
      c = chromo[3:]
    else:
      c = chromo

    if c.split('.')[-1].upper() in ('A','B'):
      try:
        key = ('%09d' % int(c.split('.')[0]), c.split('.')[-1])
      except ValueError as err:
        key = (c, c.split('.')[-1],)

    else:
      try:
        key = '%09d' % int(c)
      except ValueError as err:
        key = c

    chromos.append((key, chromo))

  if skipped:
    util.info('Skipped {:,} small chromosomes/contigs < {:,} bp'.format(len(skipped), min_contig_size))

  chromos.sort()
  chromos = [x[1] for x in chromos]

  # Get chromosome matrix index ranges
  grid = []
  chromo_offsets = {}
  chromo_spans = {}
  n = 0
  for chromo in chromos: # In display order
    s, e = chromo_limits[chromo]
    a = int(s/bin_size)
    b = int(ceil(e/float(bin_size)))
    span = b-a
    chromo_offsets[chromo] = s, n # Start bp, start bin index
    chromo_spans[chromo] = span
    n += span
    n += 1 # Add space between chromos on matrix
    grid.append(n)# At chromosome edge

  if grid:
    grid.pop() # Don't need last edge
  
  # Fill contact map matrix, last dim is for (un)ambigous
  data = np.zeros((n, n, 2), float)

  util.info('Contact map size %d x %d' % (n, n))

  if file_bin_size:
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

        contact_matrix = contacts.get((chr_a, chr_b)).astype(float)

        if contact_matrix is None: # Nothing for this pair: common for single-cell Hi-C
          continue
          
        count = int(contact_matrix.sum())
        
        bp_a, bin_a = chromo_offsets[chr_a]
        bp_b, bin_b = chromo_offsets[chr_b]
        
        size_a = chromo_spans[chr_a]
        size_b = chromo_spans[chr_b]

        sub_mat = _downsample_matrix(contact_matrix, (size_a, size_b))
        
        data[bin_a:bin_a+size_a,bin_b:bin_b+size_b,0] += sub_mat
        data[bin_b:bin_b+size_b,bin_a:bin_a+size_a,0] += sub_mat.T
        
        if chr_a != chr_b:
          if ('.' in chr_a) and ('.' in chr_b) and (chr_a.split('.')[0] == chr_b.split('.')[0]):
            n_homolog += count

          else:
            n_trans += count

        else:
          n_cis += count
        
        n_cont += count 
      
  else:
    groups = defaultdict(int)
    
    for key in contacts:
      for p_a, p_b, ag in contacts[key]:
        groups[ag] += 1

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

        s_a, n_a = chromo_offsets[chr_a]
        s_b, n_b = chromo_offsets[chr_b]
 
        for p_a, p_b, ag in contact_list:
          if chr_a != chr_b:
            if ('.' in chr_a) and ('.' in chr_b) and (chr_a.split('.')[0] == chr_b.split('.')[0]):
              homolog_groups.add(ag)

            else:
              trans_groups.add(ag)

          else:
            cis_groups.add(ag)

          a = n_a + int((p_a-s_a)/bin_size)
          b = n_b + int((p_b-s_b)/bin_size)
 
          k = 0 if groups[ag] == 1 else 1

          data[a, b, k] += 1.0
          data[b, a, k] += 1.0

    trans_groups -= homolog_groups
    cis_groups -= homolog_groups
    cis_groups -= trans_groups

    n_ambig = len([x for x in groups.values() if x > 1])
    n_homolog = len(homolog_groups)
    n_trans = len(trans_groups)
    n_cis = len(cis_groups)
    n_cont = len(groups)

  f_cis = 100.0 * n_cis / float(n_cont or 1)
  f_trans = 100.0 * n_trans / float(n_cont or 1)
  
  if n_homolog:
    f_homolog = 100.0 * n_homolog / float(n_cont or 1)  
    stats_text1 = 'Contacts:{:,d} cis:{:,d} ({:.2f}%) trans:{:,d} ({:.2f}%) homolog:{:,d} ({:.2f}%)'
    stats_text1 = stats_text1.format(n_cont, n_cis, f_cis, n_trans, f_trans, n_homolog, f_homolog)
  
  else:
    stats_text1 = 'Contacts:{:,d} cis:{:,d} ({:.2f}%) trans:{:,d} ({:.2f}%)'
    stats_text1 = stats_text1.format(n_cont, n_cis, f_cis, n_trans, f_trans)
 

  data = np.log10(data+1.0)
  
  chromo_labels = []
  for chromo in chromos:
    pos = chromo_offsets[chromo][1] + chromo_spans[chromo]/2

    if chromo.upper().startswith('CHR'):
      chromo = chromo[3:]

    chromo_labels.append((pos, chromo))
  
  if white_bg:
    cmap = LinearSegmentedColormap.from_list(name='W', colors=['#FFFFFF', '#BBBB00', '#AA0000', '#000000'], N=255)
  
  else:
    cmap = LinearSegmentedColormap.from_list(name='B', colors=['#000000', '#AA0000', '#FFFF00', '#FFFFFF'], N=255)
  
  fig, ax = plt.subplots()
  
  label_pos, labels = zip(*chromo_labels)
  
  grid = np.array(grid)
  cax = ax.matshow(data[:,:,0], cmap=cmap,origin='upper')
  ax.set_xticklabels(labels)
  ax.xaxis.set_ticks(label_pos)
  ax.xaxis.tick_bottom()
  ax.xaxis.set_tick_params(direction='out')
  ax.yaxis.set_ticks(label_pos)
  ax.yaxis.set_tick_params(direction='out')
  ax.set_yticklabels(labels)
  ax.set_title(os.path.basename(in_path))
  ax.text(0, -14, stats_text1)
  
  cbar = plt.colorbar(cax)
  cbar.set_label('$log_{10} (count+1)$')
  
  if screen_gfx:
    plt.show()
  else:
    plt.savefig(out_path)
    util.info('Written {}'.format(out_path))


def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)

  arg_parse.add_argument(metavar='CONTACT_FILE', nargs='+', dest='i',
                         help='Input NCC format (single-cell) or NPZ (binned, bulk Hi-C data) chromatin contact file(s). Wildcards accepted')

  arg_parse.add_argument('-o', metavar='OUT_FILE', nargs='+', default=None,
                         help='Optional output file name. If not specified, a default based on the input file name and output format will be used. ' \
                              'If multiple input contact files are specified there must be one output for each input')
  
  arg_parse.add_argument('-g', default=False, action='store_true',
                         help='Display graphics on-screen using matplotlib, where possible and do not automatically save output.')

  arg_parse.add_argument('-f', default=DEFAULT_FORMAT, metavar='OUTPUT_GFX_FORMAT',
                         help='Graphical format for output files. Default: {}. Available: {}'.format(DEFAULT_FORMAT, ', '.join(sorted(OUT_FORMATS))))

  arg_parse.add_argument('-s', default=0.0, metavar='BIN_SIZE', type=float,
                         help='Sequence region size represented by each small square (the resolution) in megabases. Default is to use a bin size that gives approx. 1000 x 1000 bins')

  arg_parse.add_argument('-m', default=0.0, metavar='MIN_CONTIG_SIZE', type=float,
                         help='The minimum chromosome/contig sequence length in Megabases for inclusion. Default is 10% of the largest chromosome/contig length.')

  arg_parse.add_argument('-w', default=False, action='store_true',
                         help='Specifies that the contact map should have a white background (default is black)')
                         
  args = vars(arg_parse.parse_args(argv))

  in_paths = args['i']
  out_paths = args['o']
  screen_gfx = args['g']
  out_format = args['f']
  bin_size = args['s']
  min_contig_size = args['m']
  white_bg = args['w']

  if not in_paths:
    arg_parse.print_help()
    sys.exit(1)
  
  if out_paths:
    if len(out_paths) != len(in_paths):
      util.fatal('The number of output file paths does not match the number input')
      
    if screen_gfx:
      util.warn('Output files will not be written in screen graphics (-g) mode')
      out_paths = [None] * len(in_paths)
      
  else:
    out_paths = [None] * len(in_paths)

  for in_path, out_path in zip(in_paths, out_paths):
    if not os.path.exists(in_path):
      util.fatal('Input contact file could not be found at "{}"'.format(in_path))

    contact_map(in_path, out_path, out_format, bin_size, screen_gfx, white_bg, min_contig_size=min_contig_size)


if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()
