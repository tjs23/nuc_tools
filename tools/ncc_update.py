import sys, os
from collections import defaultdict

PROG_NAME = 'ncc_update'
VERSION = '1.0.0'
DESCRIPTION = 'Convert NCC format Hi-C contact files from old to new versions'

def update_ncc(ncc_in, ncc_out):

  from nuc_tools import util, io
  
  group_sizes = defaultdict(int)
  
  with io.open_file(ncc_in) as in_file_obj:
    for i, line in enumerate(in_file_obj):
      ag = line.split()[12]
      
      if '.' in ag:
        msg = 'Contact data in file {} already appears to be in the lastest NCC format. Nothing to do.'
        util.critical(msg.format(ncc_in))
      
      group_sizes[ag] += 1
  
  group_lines = defaultdict(list)
  
  with io.open_file(ncc_in) as in_file_obj, io.open_file(ncc_out, 'w') as out_file_obj:
    write = out_file_obj.write
    
    for i, line in enumerate(in_file_obj):
      row = line.split()
      ag = row[12]
      size = group_sizes[ag]
      
      if ag in group_lines:
        row[12] = '0.1'
      else: # First
        row[12] = '%d.1' % size
       
      group_lines[ag].append(' '.join(row) + '\n')
      
      if len(group_lines[ag]) == size:
        for line2 in group_lines[ag]:
          write(line2)
 
  util.info('Output {:,} lines to file {} corresponding to {:,} ambiguity/read groups'.format(i+1, ncc_out, len(group_sizes)))
  

def main(argv=None):
  
  from argparse import ArgumentParser
  from nuc_tools import util, io

  if argv is None:
    argv = sys.argv[1:]
  
  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'
  arg_parse = ArgumentParser(prog='nuc_tools ' + PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(nargs=1, metavar='IN_NCC_FILE', dest='ncc_in',
                         help='Input NCC format file containing Hi-C contact data')

  arg_parse.add_argument(nargs=1, metavar='OUT_NCC_FILE', dest='ncc_out',
                         help='Output NCC format file to write reformatted contact data to')

  args = vars(arg_parse.parse_args(argv))
  
  ncc_in = args['ncc_in'][0]
  ncc_out = args['ncc_out'][0]
  
  invalid_msg = io.check_invalid_file(ncc_in)
  
  if invalid_msg:
    util.critical(invalid_msg)
  
  if io.is_same_file(ncc_in, ncc_out):
    util.critical('Input file cannot be the same as the output file')
  
  update_ncc(ncc_in, ncc_out)
  
if __name__ == '__main__':

  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
  main()


