import sys
import nuc_adapt
import nuc_ab_kmeans

PROG_NAME = 'nuc_tools'
VERSION = '0.0.1'
DESCRIPTION = 'Informatics tools for geneome structure and Hi-C analysis'
EPILOG = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'
TOOLS = {'adapt': nuc_adapt, 'ab_kmeans': nuc_ab_kmeans}

def main(argv=None):
  from argparse import ArgumentParser
  import nuc_util as util

  if argv is None:
    argv = sys.argv[1:]

  tools = sorted(TOOLS)
  sub_commands = ', '.join(tools)
  
  if not argv or argv[0] not in TOOLS:

    arg_parse = ArgumentParser(prog='nuc_tools', description=DESCRIPTION,
                              epilog=EPILOG, add_help=True)
 
    arg_parse.add_argument('cmd', metavar='TOOL_NAME', choices=tools,
                            help='The tool to run. Available: %s' % sub_commands)

    arg_parse.add_argument('remainder', nargs='*', metavar='ARGUMENT',
                            help='Arguments for the specified tool')
 
    args = vars(arg_parse.parse_args(argv))
    
  module = TOOLS[argv[0]]  
  module.main(argv[1:])
  

if __name__ == '__main__':
  main()
