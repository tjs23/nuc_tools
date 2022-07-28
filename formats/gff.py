import numpy as np

from collections import defaultdict
from nuc_tools import io
from nuc_tools import util


FEATURE_VALUES = {'gene':0.2,'mRNA':0.3,'exon':0.7,'CDS':1.0}

def get_feature_count(file_path):

  features = defaultdict(int)
  
  for line in io.open_file(file_path):
    if line[0] == '#':
      continue
    
    features[line.split('\t')[2]] += 1
    
  return features
  

def load_gene_dict(file_path): # V3

  sep1 = ';' # v3
  sep2 = '='
  
  chromo_gene_dict = {}
  gene_parent_dict = {}
  
  with io.open_file(file_path) as file_obj:
    
    for line in file_obj:
      if line[0] == '#':
        continue
 
      data = line[:-1].split('\t')
      n = len(data)
 
      if n < 8:
        continue
 
      chromo, source, feat, start, end, score, strand, frame = data[:8]  
      if feat not in ('gene','mRNA','CDS','exon'):
        continue
      
      start = int(start)
      end = int(end)  
      
      att_dict = dict([x.split(sep2) for x in data[8].split(sep1) if x])
      rid = att_dict['ID']
            
      if chromo not in chromo_gene_dict:
        chromo_gene_dict[chromo] = {}
      
      gene_dict = chromo_gene_dict[chromo]
      
      if feat == 'gene':
        gene_dict[rid] = [att_dict['Name'], start, end, strand, {}] # 
        gene_parent_dict[rid] = rid
      
      else:
        parent = att_dict['Parent']
      
        if parent in gene_parent_dict:
          gid = gene_parent_dict[parent]
          
          if feat == 'mRNA':
            gene_parent_dict[rid] = gid
          
          subdict = gene_dict[gid][-1]
          
          if feat in subdict:
            subdict[feat].append((start, end, strand))
          else:
            subdict[feat] = [(start, end, strand)]
          
        else:
          #util.warn(f'GFF {feat} feature {rid} missing parent')
          #print(line)
          continue

  return chromo_gene_dict     
   
        
def load_data_track(file_path, features=None, merge=False):
  # Should work with GFF and GTF
  # returns several data dicts, one for each type of feature ( all features if feature=None)
  
  sep1 = ';' # v3
  sep2 = '='
  
  data_dicts = {}
  chromo_map = {}
  
  with io.open_file(file_path, partial=True) as file_obj:
  
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

  with io.open_file(file_path) as file_obj:
    
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
      
      label = feat + ';'
      
      if n > 8 and sep1:
        attribs = [x for x in data[8].split(sep1) if x]
        ddict = dict([a.split(sep2) for a in attribs])
        
        if 'Name' in ddict:
          label += ddict['Name']
      
      if merge:
        feat = 'gene_features'
             
      if score == '.':
        score = 1.0
        val = 1.0
      else:
        score = float(score)
        val = score/1000.0
      
      if feat not in data_dicts:
        data_dicts[feat] = defaultdict(set)
  
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
    data_dicts[feat] = util.finalise_data_track(data_dict)
    #for chromo in data_dicts[feat]:
    #   print(chromo, feat, data_dicts[feat][chromo].shape)
  
  if merge:
    return data_dicts['gene_features']
    
  else:
    return data_dicts
  
