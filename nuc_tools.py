import sys, importlib
from core import nuc_util as util
from core import nuc_io as io
from core import nuc_parallel as parallel

import formats
import tools

PROG_NAME = 'nuc_tools'
VERSION = '0.0.1'
DESCRIPTION = 'Informatics tools for genome structure and Hi-C analysis'
EPILOG = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'
TOOLS = {'adapt': 'tools.nuc_adapt',
         'ab_kmeans': 'tools.nuc_ab_kmeans',
         'contact_map': 'tools.contact_map',
         'contact_compare': 'tools.contact_compare',
         'contact_insulation': 'tools.contact_insulation',
         'contact_probability': 'tools.contact_probability',
         #'contact_combine': 'tools.contact_combine',
         'contact_pair_points': 'tools.contact_pair_points',
         'ncc_filter': 'tools.ncc_filter',
         'ncc_bin': 'tools.ncc_bin',
         'cs_process': 'tools.cs_process',
         'structure_data_density': 'tools.structure_data_density',
         'data_track_filter': 'tools.data_track_filter',
         }

def main(argv=None):
  from argparse import ArgumentParser

  if argv is None:
    argv = sys.argv[1:]

  tools = sorted(TOOLS)
  sub_commands = ', '.join(tools)
  
  if not argv or argv[0] not in TOOLS:
    
    msg = 'AVAILABLE TOOLS: %s' % ' '.join(sorted(TOOLS))
    print(msg)

    arg_parse = ArgumentParser(prog='nuc_tools', description=DESCRIPTION,
                              epilog=EPILOG, add_help=True)
 
    arg_parse.add_argument('cmd', metavar='TOOL_NAME', choices=tools,
                            help='The tool to run. Available: %s' % sub_commands)

    arg_parse.add_argument('remainder', nargs='*', metavar='ARGUMENT',
                            help='Arguments for the specified tool')
 
    args = vars(arg_parse.parse_args(argv))
    
  module_name = TOOLS[argv[0]]
  module = importlib.import_module(module_name)
  module.main(argv[1:])
  

if __name__ == '__main__':
  main()
