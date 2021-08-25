import sys, os
from collections import defaultdict

PROG_NAME = 'ncc_filter'
VERSION = '1.1.0'
DESCRIPTION = 'Filter NCC format Hi-C contact files'

def filter_ncc(ncc_in, ncc_out, keep_cis, keep_cis_near, keep_cis_far, keep_trans,
               keep_ambig, keep_unambig, keep_homolog, keep_nonhomolog, keep_chromos,
               bp_range, min_size):

  from nuc_tools import util, io

  ag_sizes = defaultdict(int)
  ags_out = set()
  ags_out_add = ags_out.add
  
  found_chromos = set()
  found_chromos_add = found_chromos.add
  
  if keep_cis_near:
    keep_cis_near = int(keep_cis_near * 1000)
  
  if keep_cis_far:
    keep_cis_far = int(keep_cis_far * 1000)
  
  if keep_chromos:
    keep_chromos = set(keep_chromos)
  
  if bp_range:
    bp_min, bp_max = sorted(bp_range)
  
  min_size *= 1000 # Was Kb
  max_pos = defaultdict(int)
  
  util.info(f'Pre-parsing {ncc_in}')
  
  ag = 0
  with io.open_file(ncc_in) as in_file_obj:
    for i, line in enumerate(in_file_obj):
      chr_a, start_a, end_a, f_start_a, f_end_a, strand_a, \
        chr_b, start_b, end_b, f_start_b, f_end_b, strand_b, \
        ambig_group, pair_id, swap_pair = line.split()

      if ambig_group[0] != '0':
        ag += 1
        ag_sizes[ag] = int(ambig_group.split('.')[0])     
      
      if min_size:
        pos_a = int(f_start_a) if strand_a == '+' else int(f_end_a)
        pos_b = int(f_start_b) if strand_b == '+' else int(f_end_b)
 
        if pos_a > max_pos[chr_a]:
          max_pos[chr_a] = pos_a

        if pos_b > max_pos[chr_b]:
          max_pos[chr_b] = pos_b
             
      if i % 100000 == 0:
         util.info(f' .. {i:,} lines', line_return=True)
  
  util.info(f' .. {i:,} lines')
  
  util.info(f'Filtering {ncc_in}')
         
  i = 0
  j = 0
  ag = 0
  
  with io.open_file(ncc_in) as in_file_obj, open(ncc_out, 'w') as out_file_obj:
    
    write = out_file_obj.write
    
    for line in in_file_obj:
      i += 1
 
      if i % 100000 == 0:
         util.info(f' .. {i:,} in {j:,} out', line_return=True)
      
      chr_a, start_a, end_a, f_start_a, f_end_a, strand_a, \
        chr_b, start_b, end_b, f_start_b, f_end_b, strand_b, \
        ambig_group, pair_id, swap_pair = line.split()
     
      if ambig_group[0] != '0':
        ag += 1
      
      """ E.g.
      contig_18837 41004 40867 40852 40869 -
      contig_25335 39867 39792 39766 39794 -
      1.1 503 1
      """
      
      if min_size:
        if max_pos[chr_a] < min_size:
          continue

        if max_pos[chr_b] < min_size:
          continue

      if keep_chromos:
        found_chromos_add(chr_a)
        found_chromos_add(chr_b)
        
        if chr_a not in keep_chromos:
          continue
        
        if chr_b not in keep_chromos:
          continue
      
      if keep_unambig and ag_sizes[ag] > 1:
        continue
      
      if keep_ambig and ag_sizes[ag] < 2:
        continue

      if strand_a == '+':
        pos_a = int(f_start_a)
      else:
        pos_a = int(f_end_a)

      if strand_b == '+':
        pos_b = int(f_start_b)
      else:
        pos_b = int(f_end_b)
        
      if bp_range:
        if not (bp_min <= pos_a <= bp_max):
          continue

        if not (bp_min <= pos_b <= bp_max):
          continue
        
      if chr_a == chr_b:

        if keep_cis_near or keep_cis_far:
          delta = abs(pos_a-pos_b)
          
          if keep_cis_near and delta <= keep_cis_near:
            j += 1
            write(line)
            ags_out_add(ag)
            continue
            
          if keep_cis_far and delta > keep_cis_far:
            j += 1
            write(line)
            ags_out_add(ag)
            continue
        
        elif keep_cis:
          j += 1
          write(line)
          ags_out_add(ag)
          continue
        
      else:        
        
        if keep_homolog:
          if chr_a.split('.')[0] == chr_b.split('.')[0]: # chr1.a matches chr1, chr1.b etc
            j += 1
            write(line)
            ags_out_add(ag)
            continue

        elif keep_nonhomolog:
          if chr_a.split('.')[0] != chr_b.split('.')[0]:
            j += 1
            write(line)
            ags_out_add(ag)
            continue         
          
        elif keep_trans:
          j += 1
          write(line)
          ags_out_add(ag)
          continue
  
  util.info(' .. done')
  
  if keep_chromos:
    unmatched_chromos = keep_chromos - found_chromos
 
    if unmatched_chromos:
      util.warn('No contacts found for chromosome(s): %s' % (' '.join(sorted(unmatched_chromos))))

  util.info('Output {:,} lines ({:,} contacts) from input total of {:,} lines ({:,} contacts)'.format(j, len(ags_out), i, len(ag_sizes)))
  

def main(argv=None):
  
  from argparse import ArgumentParser
  from nuc_tools import util, io

  if argv is None:
    argv = sys.argv[1:]
  
  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'
  arg_parse = ArgumentParser(prog='nuc_tools ' + PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(nargs=1, metavar='NCC_FILE', dest='ncc_in',
                         help='Input NCC format file containing Hi-C contact data')

  arg_parse.add_argument(nargs=1, metavar='NCC_FILE', dest='ncc_out',
                         help='Output NCC format file to write filtered contact data to')

  arg_parse.add_argument('-a', '--keep-ambig', default=False, action='store_true', dest='a',
                         help='Keep only ambiguous/multi-position mapping contacts')

  arg_parse.add_argument('-u', '--keep-unambig', default=False, action='store_true', dest='u',
                         help='Keep only unique/unambiguous/single-position mapping contacts')

  arg_parse.add_argument('-c', '--keep-cis', default=False, action='store_true', dest='c',
                         help='Include intra-chromosomal (cis) mapping contacts')

  arg_parse.add_argument('-t', '--keep-trans', default=False, action='store_true', dest='t',
                         help='Include inter-chromosomal (trans) mapping contacts')

  arg_parse.add_argument('-m', '--min-chr-size', type=int, metavar='KB_SEPARATION', dest='m', default=0,
                         help='Minimum chromosome/contig size, in Kb, for inclusion')

  arg_parse.add_argument('-cn', '--keep-cis-near', type=int, metavar='KB_SEPARATION', dest='cn',
                         help='Include near intra-chromosomal (cis) mapping contacts within a specified (kb) sequence separation threshold')
                         
  arg_parse.add_argument('-cf', '--keep-cis-far', type=int, metavar='KB_SEPARATION', dest='cf',
                         help='Include far intra-chromosomal (cis) mapping contacts greater than a specified (kb) sequence separation threshold')

  arg_parse.add_argument('-hc', '--keep-homolog', default=False, action='store_true', dest='hc',
                         help='Include contacts between homologous chromosomes')

  arg_parse.add_argument('-nh', '--keep-non-homolog', default=False, action='store_true', dest='nh',
                         help='Include contacts between non-homologous chromosomes')

  arg_parse.add_argument('-chromo', '--keep-chromosomes', nargs='+',  dest='chromo',
                         help='Include only contacts between/within the specified chromosomes')

  arg_parse.add_argument('-bp', '--basepair-range', type=int, metavar='BASEPAIR', nargs=2,  dest='bp',
                         help='Include only contacts within the specified range of basepair positions (considering all selected chromosomes)')

  args = vars(arg_parse.parse_args(argv))
  
  
  ncc_in = args['ncc_in'][0]
  ncc_out = args['ncc_out'][0]
  
  invalid_msg = io.check_invalid_file(ncc_in)
  
  if invalid_msg:
    util.critical(invalid_msg)
  
  if io.is_same_file(ncc_in, ncc_out):
    util.critical('Input file cannot be the same as the output file')
  
  keep_cis         = args['c']
  keep_cis_near    = args['cn']
  keep_cis_far     = args['cf']
  keep_trans       = args['t']
  keep_ambig       = args['a']
  keep_unambig     = args['u']
  keep_homolog     = args['hc'] 
  keep_nonhomolog  = args['nh'] 
  keep_chromos     = args['chromo']
  bp_range         = args['bp']
  min_size         = args['m']
  
  if keep_ambig and keep_unambig:
    util.warning('Having both -a and -u options does nothing')
    keep_ambig = None
    keep_unambig = None
  
  if keep_cis and keep_cis_near:
    keep_cis = None
    util.warning('Option -c is redundant when using more specific option -cn')

  if keep_cis and keep_cis_far:
    keep_cis = None
    util.warning('Option -c is redundant when using more specific option -cf')

  if keep_homolog and keep_nonhomolog:
    util.warning('Having both -hc and -nh options is equivalent to -t alone')
    keep_homolog = None
    keep_nonhomolog = None
    keep_trans = True
  
  else:
    if keep_trans and keep_homolog:
      keep_trans = None
      util.warning('Option -t is redundant when using more specific option -hc')

    if keep_trans and keep_nonhomolog:
      keep_trans = None
      util.warning('Option -t is redundant when using more specific option -nh')
  
  if not (keep_cis or keep_cis_near or keep_cis_far or keep_trans or keep_homolog or keep_nonhomolog):
    keep_cis = True
    keep_trans = True
  
  filter_ncc(ncc_in, ncc_out, keep_cis, keep_cis_near, keep_cis_far, keep_trans,
             keep_ambig, keep_unambig, keep_homolog, keep_nonhomolog, keep_chromos,
             bp_range, min_size)
  
if __name__ == '__main__':

  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
  main()


