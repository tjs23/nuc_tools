import numpy as np

from collections import defaultdict
from core.nuc_io import open_file, DATA_TRACK_TYPE

VS_TEMPLATE = 'variableStep chrom=chr%s span=%d\n%d %.3f\n'

def save_data_track(file_path, data_dict):

  with open_file(file_path, 'w') as file_obj:
    write = file_obj.write
    write('track type=wiggle_0 name="nuc_tools_export"\n')
  
    for chromo in data_dict:
      for start, end, strand, val, orig_val, label in data_dict[chromo]:
        write(VS_TEMPLATE % (chromo, end-start, start, val))


def _get_param_dict(line):
  """ Could be used for track lines"""
  PARAM_PATT_MATCH = re.compile('\S*\s*(\S+)=(.+?)(\s+\S+=.+|\n|$)').match

  param_dict = {}
  match_obj = PARAM_PATT_MATCH(line)
  
  while match_obj:
    key, val, line = match_obj.groups()
    
    if val[0] in '\'"':
      val = val[1:-1]
    
    param_dict[key] = val
    match_obj = PARAM_PATT_MATCH(line)
  
  return param_dict
  

def load_data_track(file_path):
  
  data_dict = defaultdict(set)
    
  with open_file(file_path) as file_obj:
 
    is_fixed = False
    span = 1
    step = 1
    
    def_line = file_obj.readline()
    
    if not def_line.startswith('track '):
      file_obj.seek(0)
    
    for line in file_obj:
      
      if line.startswith('variableStep'):
        is_fixed = False
        param_dict = dict((pair.split('=') for pair in line.split()[1:]))
        chromo = param_dict['chrom']
        span = int(param_dict.get('span', 1))
 
      elif line.startswith('fixedStep'):
        is_fixed = True
        param_dict =dict((pair.split('=') for pair in line.split()[1:]))
        chromo = param_dict['chrom']
        pos = int(param_dict['start'])
        step = int(param_dict['step'])
        span = int(param_dict.get('span', 1))
 
      else:
        data = line.split()
 
        if is_fixed:
          if len(data) != 1:
            continue
 
          val = float(data[0])
          data_dict.add((pos, pos+span, 1, val, val, ''))
          pos += step
 
        else:
          if len(data) != 2:
            continue
 
          pos, val = data
          pos = int(pos)
          val = float(val)
          data_dict.add((pos, pos+span, 1, val, val, ''))
         
  for chromo in data_dict:
    data_dict[chromo] = np.array(sorted(data_dict[chromo]), dtype=DATA_TRACK_TYPE)
    # Enforce sorted by start pos : using set removes duplicates

  return dict(data_dict)

