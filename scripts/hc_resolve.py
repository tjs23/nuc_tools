import os

from math import log, exp

from time import time

import numpy as np

from collections import defaultdict

MIN_SEP = 100
SEP_THRESHOLD = 4e7
SEP_SCALE = 0.5 * SEP_THRESHOLD

def _get_network_score(chr_a, chr_b, pos_a, pos_b, bin_a, bin_b,
                       unambig_bins, chromo_bins, primary_limit=8.0, secondary_limit=0.0):
  
  plim = 10.0 ** primary_limit
  score_l = 1.0
  score_u = 1.0
  unambig_bins_get = unambig_bins.get
  
  a_bins = (bin_a-1, bin_a, bin_a+1)
  b_bins = (bin_b-1, bin_b, bin_b+1)
  
  for a in a_bins:
    rev = a < bin_a
    peri_a = a != bin_a
    
    for b in b_bins:
      key2 = (chr_a, chr_b, a, b)
      
      unambig = unambig_bins_get(key2)
      
      if unambig:
        if rev:
          unambig = unambig[::-1]
        
        if bin_a == a and bin_b == b:
          for pos_1, pos_2 in unambig:
            delta_1 = pos_1-pos_a
            delta_2 = pos_2-pos_b
 
            if delta_1 and delta_2:
              if delta_1 < 0.0:
                delta_1 = max(-delta_1, MIN_SEP)/SEP_SCALE
                score_l += exp(-delta_1*delta_1)
              else:
                delta_1 = max(delta_1, MIN_SEP)/SEP_SCALE
                score_u += exp(-delta_1*delta_1)

              if delta_2 < 0.0:
                delta_2 = max(-delta_2, MIN_SEP)/SEP_SCALE
                score_l += exp(-delta_2*delta_2)
              else:
                delta_2 = max(delta_2, MIN_SEP)/SEP_SCALE
                score_u += exp(-delta_2*delta_2)
             
         
        else:
          for pos_1, pos_2 in unambig:
            delta_1 = pos_1-pos_a
            delta_2 = pos_2-pos_b
 
            if (abs(delta_1) < SEP_THRESHOLD) and (abs(delta_2) < SEP_THRESHOLD):
 
              if delta_1 < 0.0:
                delta_1 = max(-delta_1, MIN_SEP)/SEP_SCALE
                score_l += exp(-delta_1*delta_1)
              else:
                delta_1 = max(delta_1, MIN_SEP)/SEP_SCALE
                score_u += exp(-delta_1*delta_1)

              if delta_2 < 0.0:
                delta_2 = max(-delta_2, MIN_SEP)/SEP_SCALE
                score_l += exp(-delta_2*delta_2)
              else:
                delta_2 = max(delta_2, MIN_SEP)/SEP_SCALE
                score_u += exp(-delta_2*delta_2)
 
            elif peri_a and abs(delta_1) > SEP_THRESHOLD:
              break
        
        if score_l * score_u > plim:
          return primary_limit
        
  if secondary_limit and log(score_l * score_u, 10.0) < secondary_limit: # If not goot enough, check deeper
    intermed_a = defaultdict(list)
    intermed_b = defaultdict(list)
 
    for chr_c, bin_c in chromo_bins:
      append = intermed_a[(chr_c, bin_c)].append
 
      for a in range(bin_a-1, bin_a+2):
        if chr_a > chr_c:
          i, j = 1, 0
          key2 = (chr_c, chr_a, bin_c, a)

        else:
          i, j = 0, 1
          key2 = (chr_a, chr_c, a, bin_c)
 
        for row in unambig_bins_get(key2, []):
          delta = abs(row[i]-pos_a)
 
          if delta and delta < SEP_THRESHOLD:
            append((row[j], row[i]-pos_a))
 
    for chr_c, bin_c in intermed_a:
      append = intermed_b[(chr_c, bin_c)].append
 
      for b in range(bin_b-1, bin_b+2):
        if chr_b > chr_c:
          i, j = 1, 0
          key2 = (chr_c, chr_b, bin_c, b)

        else:
          i, j = 0, 1
          key2 = (chr_b, chr_c, b, bin_c)
 
        for row in unambig_bins_get(key2, []):
          delta = abs(row[i]-pos_b)
 
          if delta and delta < SEP_THRESHOLD:
            append((row[j], row[i]-pos_b))
 
    for key2, values in intermed_b.items():
      for pos_2, delta_2 in values:
        for pos_1, delta_1 in intermed_a[key2]:
          delta_3 = abs(pos_1-pos_2)
 
          if delta_3 and delta_3 < SEP_THRESHOLD:
            for d in (delta_1, delta_2, delta_3):

              if d < 0.0:
                d = max(-d, MIN_SEP)/SEP_SCALE
                score_l += exp(-d*d)
              else:
                d = max(d, MIN_SEP)/SEP_SCALE
                score_u += exp(-d*d)

  return log(score_l * score_u, 10.0)


def _write_ambig_filtered_ncc(in_file_path, out_ncc_path, ag_data, resolved_ag=None, removed_ag=None):
  
  if not resolved_ag:
    resolved_ag = {}
    
  if not removed_ag:
    removed_ag = set()
  
  n = 0
  ambig_group = 0
  seen_res_group = set()
  
  with open(out_ncc_path, 'w') as out_file_obj, open(in_file_path) as in_file_obj:
    write = out_file_obj.write

    for i, line in enumerate(in_file_obj):
      row = line.split()
      chr_a = row[0]
      chr_b = row[6]
      ambig_code = row[12]

      if chr_a > chr_b:
        chr_a, chr_b = chr_b, chr_a

      if '.' in ambig_code: # Updated NCC format
        if int(float(ambig_code)) > 0:
          ambig_group += 1
      else:
        ambig_group = int(ambig_code)    
      
      sz = len(ag_data[ambig_group])
      
      if ambig_group in resolved_ag:    
        keep = 1 if i in resolved_ag[ambig_group] else 0 # Else inactivate
        
        if ambig_group in seen_res_group:
          row[12] = '0.%d' % (keep,)
        else:
          seen_res_group.add(ambig_group)
          row[12] = '%d.%d'  % (sz,keep)
                    
        line = ' '.join(row)+'\n'

      elif ambig_group in removed_ag: # Inactivate
        if ambig_group in seen_res_group:
          row[12] = '0.0'
        else:
          seen_res_group.add(ambig_group)
          row[12] = '%d.0'  % (sz,)
           
        line = ' '.join(row)+'\n'
        
      write(line)    
      
      n += 1

  msg = "Written {:,} of {:,} lines to {}".format(n, i+1, out_ncc_path)
  
  print(msg)
   
      
def _load_bin_sort_ncc(in_ncc_path, sep_threshold=SEP_THRESHOLD):
  
  msg = 'Reading %s' % in_ncc_path
  print(msg)

  ag_data = defaultdict(list)
  nonambig_bins = defaultdict(list)
  ag_positions = defaultdict(set)
  pos_ambig = set()
  
  # Identify positional ambiguity
  
  with open(in_ncc_path) as in_file_obj:
    ambig_group = 0

    for line in in_file_obj:
      chr_a, start_a, end_a, f_start_a, f_end_a, strand_a, chr_b, start_b, end_b, \
        f_start_b, f_end_b, strand_b, ambig_code, pair_id, swap_pair = line.split()
        
      if '.' in ambig_code: # Updated NCC format
        if int(float(ambig_code)) > 0:
          ambig_group += 1 # Count even if inactive; keep consistent group numbering

      else:
        ambig_group = int(ambig_code)          

      if ambig_code.endswith('.0'): # Inactive
        continue
      
      if strand_a == '+':
        pos_a = int(f_end_a)
      else:
        pos_a = int(f_start_a)

      if strand_b == '+':
        pos_b = int(f_end_b)
      else:
        pos_b = int(f_start_b)
      
      ag_positions[ambig_group].add((chr_a, pos_a))
      ag_positions[ambig_group].add((chr_b, pos_b))
  
  n_pos_ag = 0
  for ag in ag_positions:
    chr_pos = ag_positions[ag]
    
    if len(chr_pos) > 4:
      pos_ambig.add(ag)
      
    else:
      chr_roots = set([x[0].split('.')[0] for x in chr_pos])
      
      if len(chr_roots) > 2:
        pos_ambig.add(ag)
  
  msg = 'Found {:,} positional ambiguity groups from {:,} '.format(len(pos_ambig), ambig_group)
  print(msg)
    
  # Bin and group
  ambig_bins = defaultdict(list)
  i = -1 # File could be empty
  
  with open(in_ncc_path) as in_file_obj:
    ambig_group = 0

    for i, line in enumerate(in_file_obj):
      chr_a, start_a, end_a, f_start_a, f_end_a, strand_a, chr_b, start_b, end_b, \
        f_start_b, f_end_b, strand_b, ambig_code, pair_id, swap_pair = line.split()

      if '.' in ambig_code: # Updated NCC format
        if int(float(ambig_code)) > 0:
          ambig_group += 1
      else:
        ambig_group = int(ambig_code)          
      
      if ambig_code.endswith('.0'): # Inactive
        continue
      
      if strand_a == '+':
        pos_a = int(f_end_a)
      else:
        pos_a = int(f_start_a)

      if strand_b == '+':
        pos_b = int(f_end_b)
      else:
        pos_b = int(f_start_b)
 
      if chr_a > chr_b:
        chr_a, chr_b = chr_b, chr_a
        pos_a, pos_b = pos_b, pos_a
      
      bin_a = int(pos_a/sep_threshold)
      bin_b = int(pos_b/sep_threshold)
 
      key = (chr_a, chr_b, bin_a, bin_b)
      ag_data[ambig_group].append((key, pos_a, pos_b, i))
      
      
      if ambig_group in pos_ambig: 
        ambig_bins[key].append((pos_a, pos_b, ambig_group))
      else:
        nonambig_bins[key].append((pos_a, pos_b, ambig_group))
      
      
      
  msg = 'Loaded {:,} contact pairs in {:,} ambiguity groups'.format(i+1, len(ag_data))
  print(msg)
 
  msg = ' .. sorting data'
  print(msg)

  unambig_bins = {}
  chromo_pair_counts = defaultdict(int)
  chromo_bins = set()
  chromos = set()
  
  for key in nonambig_bins:
    chr_a, chr_b, bin_a, bin_b = key
    chromos.update((chr_a, chr_b))
    vals = sorted(nonambig_bins[key]) # Sort by 1st chromo pos
    unambig = [x[:2] for x in vals if len(ag_data[x[2]]) == 1]
 
    unambig_bins[key] = unambig
    nonambig_bins[key] = vals
             
    if unambig:
      chromo_bins.add((chr_a, bin_a))
      chromo_bins.add((chr_b, bin_b))
      
      chromo_pair_counts[(chr_a, chr_b)] += len(unambig)
      
  msg = ' .. done'
  print(msg)
 
  return sorted(chromos), ag_data, pos_ambig, chromo_bins, nonambig_bins, unambig_bins, ambig_bins, dict(chromo_pair_counts)


def remove_isolated_unambig(in_ncc_path, out_ncc_path, threshold=0.01, sep_threshold=SEP_THRESHOLD, homo_trans_dens_quant=90.0):
  
  chromos, ag_data, pos_ambig, chromo_bins, nonambig_bins, unambig_bins, ambig_bins, chromo_pair_counts = _load_bin_sort_ncc(in_ncc_path)
 
  msg_template = ' .. Processed:{:>7,} Removed:{:>7,}'
  
  all_bins = {}
  for key in set(ambig_bins) | set(nonambig_bins):
    all_bins[key] = ambig_bins.get(key, []) + nonambig_bins.get(key, [])
  
  removed_ag = set()
  resolved_ag = {}
  
  msg = 'Removing inter-homologue contacts from anomolously dense regions'
  print(msg)
  
  bin_counts = {}
  for key in all_bins:
    chr_a, chr_b, bin_a, bin_b = key
    root_a = chr_a.split('.')[0]
    root_b = chr_b.split('.')[0]
    
    if root_a == root_b:
      n_contacts = len(all_bins[key])
      
      if n_contacts:
        bin_counts[key] = n_contacts
  
  if not bin_counts:
    return 0
  
  counts = list(bin_counts.values())
  upper_dens_thresh = np.percentile(counts, homo_trans_dens_quant)
  
  """
  print('Quantiles', np.log10(np.percentile(counts, [1,5,10,50,80])))
  print('Quantiles', np.percentile(counts, [1,5,10,50,80]))
  from matplotlib import pyplot as plt
  plt.hist(np.log10(counts), bins=100)
  plt.show()     
  """
  
  for ag in ag_data:
    pairs = ag_data[ag]
    n_pairs = len(pairs)
    keep = []
    
    for j, (key, pos_a, pos_b, line_idx) in enumerate(pairs):
      chr_a, chr_b, bin_a, bin_b = key
 
      if (chr_a != chr_b) and (key in bin_counts): # Homolog trans only
        count = bin_counts[key]
        
        if count < upper_dens_thresh:
          for a in range(bin_a-1, bin_a+2):
            for b in range(bin_b-1, bin_b+2):
              if a == bin_a and b == bin_b:
                continue
 
              key2 =  (chr_a, chr_b, a, b)
 
              for pos1, pos2, ag1 in all_bins.get(key2, []):
                if (abs(pos1-pos_a) < sep_threshold) and (abs(pos2-pos_b) < sep_threshold):
                  count += 1
 
                  if count > upper_dens_thresh:
                    break
              
              else:
                continue
              break
            
            else:
              continue
            break  
              
        if (count < upper_dens_thresh) and (n_pairs > 1): # ONly keep ambiguous holologous trans
          keep.append(j)
        
        elif (n_pairs == 1) and (key in unambig_bins):
          
          contacts = []
          for pos1, pos2 in unambig_bins[key]:
            if (pos1 != pos_a) or (pos2 != pos_b):
              contacts.append((pos1, pos1))
          
          unambig_bins[key] = contacts
          
      else:
        keep.append(j)
    
    n_keep = len(keep)
        
    if n_keep:
      if n_keep < n_pairs:
        new_pairs = [pairs[j] for j in keep]
        ag_data[ag] = new_pairs
      
        if n_keep > 1: # Still ambiguous
          resolved_ag[ag] = [x[3] for x in new_pairs] # Line indices        
    
    else:      
      removed_ag.add(ag)
    
  msg = ' .. removed {:,}'.format(len(removed_ag))
  print(msg)
 
  msg = ' .. partly resolved {:,}'.format(len(resolved_ag))
  print(msg)

  scores = []
  it = 0

  unambig_pairs = [(g, ag_data[g][0]) for g in ag_data if len(ag_data[g]) == 1 and (g not in removed_ag)] #  and (g not in resolved_ag)]
 
  msg = 'Scoring {:,} unambiguous pairs'.format(len(unambig_pairs))
  print(msg)
  
  for ag, (key, pos_a, pos_b, line_idx) in unambig_pairs:

    it += 1
    if it % 1000 == 0:
      msg = msg_template.format(it, len(removed_ag))
      print(msg)
 
    group_score = 0.0 
    chr_a, chr_b, bin_a, bin_b = key
      
    contacts = unambig_bins[key]
 
    if contacts:
      score = _get_network_score(chr_a, chr_b, pos_a, pos_b, bin_a, bin_b, unambig_bins, chromo_bins) # , secondary_limit=threshold) 
    else:
      score = 0.0
     
    if score < threshold:
      removed_ag.add(ag)
  
  
  _write_ambig_filtered_ncc(in_ncc_path, out_ncc_path, ag_data, resolved_ag=resolved_ag, removed_ag=removed_ag)

  msg = msg_template.format(it, len(removed_ag))     
  print(msg)
  
  return len(ag_data)
  
  
def network_filter_ambig(ag_data, missing_cis, trans_close, nonambig_bins, unambig_bins, chromo_bins,
                         score_threshold, min_trans_relay=5, remove_isolated=False, max_pairs=16):

  msg_template = ' .. Processed:{:>7,} Resolved:{:>7,} Step_time:{:.3f} s Time taken:{:5.2f} s'
  
  resolved_ag = {}
  removed_ag = set()
  
  scores = [[], [], [], []]
  start_time = t0 = time()
  
  
  qc = []
    
  for j, ag in enumerate(ag_data):
  
    if j % 10000 == 0:
      t1 = time()
      msg = msg_template.format(j, len(resolved_ag), t1-t0, t1-start_time)
      t0 = t1
      print(msg)
    
    if ag in removed_ag:
      continue
    
    pairs = ag_data[ag]
 
    if len(pairs) == 1:
      continue
      
    if len(pairs) > max_pairs:
      removed_ag.add(ag)
      continue

    poss_pairs = []
    for key, pos_a, pos_b, line_idx in pairs:
      chr_pair = tuple(key[:2])
      
      if chr_pair in missing_cis:
        continue
        
      if (chr_pair not in trans_close) or (trans_close[chr_pair] >= min_trans_relay): 
        poss_pairs.append((key, pos_a, pos_b, line_idx))
    
    n_poss = len(poss_pairs)
    if poss_pairs and n_poss < len(pairs): # A chromo pair can be excluded
      if len(poss_pairs) == 1: # Only one sensible trans possibility
         key, pos_a, pos_b, line_idx = poss_pairs[0]
         chr_a, chr_b, bin_a, bin_b = key
          
         if nonambig_bins.get(key) and _get_network_score(chr_a, chr_b, pos_a, pos_b, bin_a, bin_b, unambig_bins, chromo_bins):
           resolved_ag[ag] = (line_idx,)
         else:
           removed_ag.add(ag)
        
         continue
      
      else:
        resolved_ag[ag] = tuple([x[3] for x in poss_pairs]) # Might be refined further

    line_indices = []
    group_scores = [0.0] * max(4, len(pairs))
    
    for p, (key, pos_a, pos_b, line_idx) in enumerate(pairs):
      line_indices.append(line_idx)
      chr_a, chr_b, bin_a, bin_b = key
      contacts = nonambig_bins.get(key)
     
      if (chr_a, chr_b) in missing_cis:
        score = 0.0
 
      elif contacts:
        score = _get_network_score(chr_a, chr_b, pos_a, pos_b, bin_a, bin_b, unambig_bins, chromo_bins)
 
        if score and chr_a != chr_b:
          qc.append(score)
 
      else:
        score = 0.0
        
      group_scores[p] = score

    a, b, c, d, *e = np.argsort(group_scores)[::-1]
    
    if group_scores[a] < 1.0: # Best is isolated
      if remove_isolated:
        if ag in resolved_ag:
          del resolved_ag[ag]
          
        removed_ag.add(ag)
     
    elif group_scores[a]  >  score_threshold * group_scores[b]:
      resolved_ag[ag] = (line_indices[a],)
      
    elif group_scores[b]  >  score_threshold * group_scores[c]: # First two were close
      resolved_ag[ag] = (line_indices[a], line_indices[b])

    elif group_scores[c]  >  score_threshold * group_scores[d]: # First three were close
      resolved_ag[ag] = (line_indices[a], line_indices[b], line_indices[c])
    
 
  #from matplotlib import pyplot as plt
  
  #fig, ax1 = plt.subplots()
  
  #ax1.hist(qc, bins=250)
  
  #plt.show()     
    
  return resolved_ag, removed_ag
  
  
  
def resolve_contacts(in_ncc_path, out_ncc_path, remove_isolated=True, score_threshold=2.0,
                     remove_pos_ambig=False, primary_weight=5, trans_relay_percentile=5.0):

  msg_template = 'Processed:{:>7,} Resolved:{:>7,}'

  msg = 'Reading contact data'
  print(msg)
  
  chromos, ag_data, pos_ambig, chromo_bins, nonambig_bins, unambig_bins, ambig_bins, chromo_pair_counts = _load_bin_sort_ncc(in_ncc_path)

  missing_cis = set()
  
  for i, chr_a in enumerate(chromos):
    for chr_b in chromos[i:]:
      n_pair = chromo_pair_counts.get((chr_a, chr_b), 0)
      
      if chr_a == chr_b:
        if n_pair < primary_weight:
          missing_cis.add((chr_a, chr_b))
          msg = f'Chromosome {chr_a} is missing!'
          print(msg)
  
  # Do a light disambiguation initially, which gives better trans relay stats
  # - min_trans_relay should come from stats

  msg = 'Primary filtering with network scores'
  print(msg)
  
  temp_ncc_path = os.path.splitext(out_ncc_path)[0] + '_temp.ncc'
  
  resolved_ag, removed_ag = network_filter_ambig(ag_data, missing_cis, {}, nonambig_bins, unambig_bins,
                                                 chromo_bins, score_threshold)

  _write_ambig_filtered_ncc(in_ncc_path, temp_ncc_path, ag_data, resolved_ag, removed_ag)
  
  chromos, ag_data, pos_ambig, chromo_bins, nonambig_bins, unambig_bins, ambig_bins, chromo_pair_counts = _load_bin_sort_ncc(temp_ncc_path)
  
  print(msg_template.format(len(ag_data), len(resolved_ag)))  
  
  trans_close = {}  
  for i, chr_a in enumerate(chromos):
    for chr_b in chromos[i:]:

      if chr_a != chr_b:
        n_pair = primary_weight * chromo_pair_counts.get((chr_a, chr_b), 0)
        
        for chr_c in chromos:
          if chr_c in (chr_a, chr_b):
            continue
 
          a = chromo_pair_counts.get((chr_a, chr_c), 0)
          b = chromo_pair_counts.get((chr_b, chr_c), 0)
          c = chromo_pair_counts.get((chr_c, chr_a), 0)
          d = chromo_pair_counts.get((chr_c, chr_b), 0)
 
          n_pair += np.sqrt((a + c) * (b + d))
 
        trans_close[(chr_a, chr_b)] = n_pair
        trans_close[(chr_b, chr_a)] = n_pair
 
  
  from matplotlib import pyplot as plt
  
  trans_vals = sorted([x for x in trans_close.values() if x])

  """
  fig, (ax1, ax2) = plt.subplots(2, 1)
  
  vals1 = sorted([x for x in trans_close0.values() if x])
  
  print(np.percentile(vals1, [1,2,5,10]))
  print(np.percentile(trans_vals, [1,2,5,10]))
  
  ax1.hist(vals1, bins=100, range=(0, 100))
  
  ax2.hist(tras_vals, bins=100, range=(0, 100))
  
  plt.show()     
  """
  
  min_trans_relay = np.percentile(trans_vals, trans_relay_percentile)


  msg = 'Secondary filtering with network scores'
  print(msg)

  chromos, ag_data, pos_ambig, chromo_bins, nonambig_bins, unambig_bins, ambig_bins, chromo_pair_counts = _load_bin_sort_ncc(in_ncc_path)

  resolved_ag, removed_ag = network_filter_ambig(ag_data, missing_cis, trans_close, nonambig_bins, unambig_bins,
                                                 chromo_bins, score_threshold, min_trans_relay, remove_isolated)  

  if remove_pos_ambig: 
    removed_ag.update(pos_ambig)
  
  _write_ambig_filtered_ncc(in_ncc_path, out_ncc_path, ag_data, resolved_ag, removed_ag)
  
  print(msg_template.format(len(ag_data), len(resolved_ag)))  


if __name__ == '__main__': 
  
  
  # All functions take an NCC file and create a filtered NCC file
  
  # Cleanup ambig so noise doesn't affect disambiguation too much
  import sys
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  
  from tools.contact_map import contact_map
  
  from glob import glob
  
  #ncc_file_paths = glob('/data/hi-c/hybrid/ncc/SR*-*[1234567890].ncc') + glob('/data/hi-c/hybrid/laue_hybrid/SLX*.ncc') + glob('/data/hi-c/hybrid/Hi-C_53_SLX-18853/SLX*.ncc')
  #ncc_file_paths = ['/data/hi-c/hybrid/Hi-C_53_SLX-18853/SLX-18853_INLINE_HJWKFDRXX_s_1_r_1_2_P83F7.ncc']
  
  ncc_file_paths = sys.argv[1:]
    
  for ncc_file_path in ncc_file_paths:
    file_root = os.path.splitext(ncc_file_path)[0]
 
    tag = file_root.split('_')[-1]
    if tag in ('clean','filter','refilter','extra'):
      continue
 
    clean_ncc_path   = file_root + '_clean.ncc'
    filter_ncc_path = file_root + '_filter.ncc'
    #extra_ncc_path = file_root + '_filter_temp.ncc'
    
    #if os.path.exists(filter_ncc_path):
    #  continue
    
    clean_pdf   = file_root + '_clean.pdf'
    filter_pdf = file_root + '_filter.pdf'
    #extra_pdf = file_root + '_filter_temp.pdf'
 
    #show_chromos = ('chr1.a', 'chr1.b', 'chr2.a', 'chr2.b')
    show_chromos = None
    
    n = remove_isolated_unambig(ncc_file_path, clean_ncc_path)
    
    if n > 1e5:
       contact_map([clean_ncc_path], clean_pdf, bin_size=None, show_chromos=show_chromos,
                    no_separate_cis=True, is_single_cell=True)
       
       # Cautious threshold
       resolve_contacts(clean_ncc_path, filter_ncc_path, score_threshold=10.0, remove_isolated=True)

       #contact_map([extra_ncc_path], extra_pdf, bin_size=None, show_chromos=show_chromos,
       #             no_separate_cis=True, is_single_cell=True)

       contact_map([filter_ncc_path], filter_pdf, bin_size=None, show_chromos=show_chromos,
                    no_separate_cis=True, is_single_cell=True)

       # More permissive threshold
       #resolve_contacts(filter_ncc_path, extra_ncc_path, score_threshold=2.0, remove_isolated=True)

 


 
"""
all ncc files (more than 40) for Nagano dip3 is on /mnt/delphi/scratch/shared/dip_ncc/dip_3

and also the report and pdf contact maps.

Sorry those perhaps include few ones that do not give good structures. If you want to know which one gives a structure, all 400k structures
are inside /mnt/delphi/scratch/shared/Nagano_serumLif/dip_3


The naming is very weird: the ones after filtering/disambiguation is called ...._filtered_isoRm_hc_remove_more.ncc sorry for a very long name
that only makes sense to me... because this indicates the version of code I'm using...

Another note: all original ncc files are in new format , i.e. the 1.1,1.2 etc column, while _filtered_isoRm_hc_remove_more.ncc files are in
old format: I copied the 14th column to the 13th column (13th and 14th are the same), there is no 1.1 like columns. That's due to historical
reasons. I coded the code in the way that it works with the old format initially. By the time Wayne was testing the new version of
nuc_dynamics, sometimes it still have problem with new format.  So I thought it was all safer to keep the old format as output (it takes new
format as input fine).

I've attached the code here for disambiguation. The way to run it is python3 hc_resolve_v3_old_hyb.py ncc_file_input 15

(set 15 for Nagano's and our new hyb, and 25 for our old hyb)(normally it's all 15, except our weird old line which has very very few unambig
contacts).

it will output some other intermediate files... delete all of them, no use.... it will also print out tons of numbers on screen, again,
ignore all of them. Those are only for some checking, would be useful in extreme cases to identify possible problems. I will make it nicer
for users once for publication...

At /mnt/delphi/wb104/nuc_dynamics2_runs/diploid_3. And /mnt/delphi \u2014> /home if you are looking on delphi itself rather than demeter
(etc.).

> On 17 Nov 2020, at 14:52, X. Ma <xm227@cam.ac.uk> wrote: > > For structurally disambiguated final ncc files, they are all inside Wayne's
folder delphi/wb104/nuc_dynamics2_runs/dip3 or dip_3--find the 400k final ncc.

delphi:/home/scratch/shared/dip_ncc/dip_3/

Processed data at:

munro-i7:/data/hi-c/hybrid/fastq


                                                                      	      	    	      	    -------- Coordinate RMSDs -------
                                                                  File	p_size	n_chr	n_coord	   mt50	    p50	     m0	    m50	   m100
    SLX-20046_INLINE_H3CYTDRXY_s_1_r_1_2_CB924h_P92J16_filter_8000.n3d	  8000	   39	    665	  0.348	  0.155	  0.085	  0.422	  0.704
    SLX-20046_INLINE_H3CYTDRXY_s_1_r_1_2_CB924h_P92J17_filter_8000.n3d	  8000	   39	    665	  1.070	  1.374	  0.130	  2.426	  4.647
    SLX-20046_INLINE_H3CYTDRXY_s_1_r_1_2_CB924h_P92I17_filter_8000.n3d	  8000	   38	    643	  1.560	  1.402	  0.198	  2.783	  4.390
SLX-20046_INLINE_H3CYTDRXY_s_1_r_1_2_CB9Nanog2iL_P91F6_filter_8000.n3d	  8000	   39	    661	  3.260	  2.410	  1.298	  4.489	  9.231
SLX-20046_INLINE_H3CYTDRXY_s_1_r_1_2_CB9Nanog2iL_P90H5_filter_8000.n3d	  8000	   38	    651	  8.922	  6.076	  5.320	 10.020	 13.532

    SLX-20046_INLINE_H3CYTDRXY_s_1_r_1_2_CB924h_P92J16_filter_8000.n3d	  8000	   39	    665	  0.348	  0.155	  0.085	  0.422	  0.704
    SLX-20046_INLINE_H3CYTDRXY_s_1_r_1_2_CB924h_P92J17_filter_8000.n3d	  8000	   40	    687	  0.630	  0.812	  0.270	  1.218	  3.950
    SLX-20046_INLINE_H3CYTDRXY_s_1_r_1_2_CB924h_P92I17_filter_8000.n3d	  8000	   40	    687	  0.767	  0.877	  0.539	  0.953	  3.731
SLX-20046_INLINE_H3CYTDRXY_s_1_r_1_2_CB9Nanog2iL_P91F6_filter_8000.n3d	  8000	   40	    687	  1.193	  1.950	  0.718	  3.636	  4.709
SLX-20046_INLINE_H3CYTDRXY_s_1_r_1_2_CB9Nanog2iL_P90H5_filter_8000.n3d	  8000	   40	    687	  1.314	  2.002	  0.938	  3.120	  5.714



"""      
      
      
      
      
      
      
      
      
