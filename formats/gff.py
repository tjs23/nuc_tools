import numpy as np

from collections import defaultdict
from nuc_tools.core.nuc_io import open_file
from nuc_tools.core.nuc_util import finalise_data_track

def get_feature_count(file_path):

  features = defaultdict(int)
  
  for line in open_file(file_path):
    if line[0] == '#':
      continue
    
    features[line.split('\t')[2]] += 1
    
  return features
  

def load_data_track(file_path, features=None):
  # Should work with GFF and GTF
  # returns several data dicts, one for each type of feature ( all features if feature=None)
  
  sep1 = ';' # v3
  sep2 = '='
  
  data_dicts = {}
  chromo_map = {}
  
  with open_file(file_path, partial=True) as file_obj:
  
    for line in file_obj:
      if line[0] == '#':
        continue
 
      data = line[:-1].split('\t')
 
      if len(data) > 8:
        attribs = data[8]
 
        if ';' in attribs:
          if '; ' in attribs:
            sep1 = '; ' # v2
 
          if '=' not in attribs.split(';')[0]:
            sep2 = None
 
        else:
          sep1 = None
 
        break

  with open_file(file_path) as file_obj:
    
    for line in file_obj:
      if line[0] == '#':
        continue
 
      data = line[:-1].split('\t')
      n = len(data)
 
      if n < 8:
        continue
 
      chromo, source, feat, start, end, score, strand, frame = data[:8]
      
      if feat == 'region':
        attribs = [x for x in data[8].split(sep1) if x]
        ddict = dict([a.split(sep2) for a in attribs])
        if 'chromosome' in ddict:
          chromo_map[chromo] = ddict['chromosome']
      
      if features and (feat not in features):
        continue
      
      if feat not in data_dicts:
        data_dicts[feat] = defaultdict(set)
      
      if n > 8 and sep1:
        attribs = [x for x in data[8].split(sep1) if x]
        ddict = dict([a.split(sep2) for a in attribs])
        label = ddict.get('Name', feat)
          
      else:
        label = feat
 
      if score == '.':
        score = 1.0
        val = 1.0
      else:
        score = float(score)
        val = score/1000.0
 
      strand = 0 if strand == '-' else 1
      data_dicts[feat][chromo].add((int(start), int(end), strand, score, val, label[:32]))
  
  for feat, data_dict in data_dicts.items():
    for chromo in sorted(data_dict):
      if chromo in chromo_map:
        chrom_name = 'chr' + chromo_map[chromo]
        
        if chrom_name not in data_dict:
          data_dict[chrom_name] = set()
        
        data_dict[chrom_name] |= data_dict[chromo]
      
  for feat, data_dict in data_dicts.items():
    data_dicts[feat] = finalise_data_track(data_dict)
     
  return data_dicts
  
