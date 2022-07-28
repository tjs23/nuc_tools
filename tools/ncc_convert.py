import sys, os
from collections import defaultdict

PROG_NAME = 'ncc_convert'
VERSION = '1.0.0'
DESCRIPTION = 'Convert NCC format Hi-C contact files to other bioinformatics formats'

FORMATS = set(['BED',])
AVAIL_FORMATS = ', '.join(sorted(FORMATS))

def convert_ncc(ncc_in, out_fmt, report_freq=100000):
  
  from nuc_tools import util, io
    
  file_root, file_ext = os.path.splitext(ncc_in)
  file_ext = out_fmt.lower()
  
  file_out = f'{file_root}.{file_ext}' 
  temp_file_out = file_out + '_ncc_conv_temp'
  
  group_sizes = defaultdict(int)
  
  out_file_obj = io.open_file(temp_file_out, 'w')
  write = out_file_obj.write
  
  mapq = 37
   
  with io.open_file(ncc_in) as in_file_obj:
    util.info(f'Reading {ncc_in}')
   
    for i, line in enumerate(in_file_obj):
      chr_a, f_start_a, f_end_a, start_a, end_a, strand_a, \
      chr_b, f_start_b, f_end_b, start_b, end_b, strand_b, \
      ambig_code, read_id, swap_pair = line.split()
      group_sizes[ambig_code] += 1
      read_name = f'READ_{int(read_id):010d}'
      
      if i % report_freq == 0:
        util.info(f' .. processed {i:,} NCC lines', line_return=True)
      
      if out_fmt == 'BED':
        write(f'{chr_a}\t{start_a}\t{end_a}\t{read_name}/1\t{mapq}\t{strand_a}\n')
        write(f'{chr_b}\t{start_b}\t{end_b}\t{read_name}/2\t{mapq}\t{strand_b}\n')
  
  util.info(f' .. processed {i:,} NCC lines', line_return=True)
  
  out_file_obj.close()
 
  util.info(f'Sorting output')
  
  cmd_args = ['sort','-k','4',temp_file_out]
  util.call(cmd_args, stdin=None, stdout=file_out, stderr=None, verbose=True, wait=True, path=None, shell=False)
  
  os.unlink(temp_file_out)
  
  util.info('Converted {:,} input lines to file {}, corresponding to {:,} ambiguity/read groups'.format(i+1, file_out, len(group_sizes)))
  

def main(argv=None):
  
  from argparse import ArgumentParser
  from nuc_tools import util, io

  if argv is None:
    argv = sys.argv[1:]
  
  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'
  arg_parse = ArgumentParser(prog='nuc_tools ' + PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(nargs=1, metavar='IN_NCC_FILE', dest='i',
                         help='Input NCC format file containing Hi-C contact data')

  arg_parse.add_argument(nargs=1, metavar='OUT_FORMAT', dest='f',
                         help=f'Output file format to write reformatted contact data to. Must be one of {AVAIL_FORMATS}.')

  args = vars(arg_parse.parse_args(argv))
  
  ncc_in = args['i'][0]
  out_fmt = args['f'][0]
  
  invalid_msg = io.check_invalid_file(ncc_in)
  
  if invalid_msg:
    util.critical(invalid_msg)
  
  out_fmt = out_fmt.upper()
  
  if out_fmt not in FORMATS:
    util.critical(f'Output format {out_fmt} is not one of {AVAIL_FORMATS}')
   
  convert_ncc(ncc_in, out_fmt)
  
  
if __name__ == '__main__':

  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
  main()


#/data/old/nucleus/processing/nuc_processing/paper_ncc/P2J8.ncc
