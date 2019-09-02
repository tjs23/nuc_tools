import os

import numpy as np

from collections import defaultdict

MIN_SEP = 100
SEP_THRESHOLD = 1e7

def _get_network_score(chr_a, chr_b, pos_a, pos_b, bin_a, bin_b,
                       unambig_bins, chromo_bins, second_level=False):
  
  unambig_bins_get = unambig_bins.get
  close = []
  
  for a in range(bin_a-1, bin_a+2):
    for b in range(bin_b-1, bin_b+2):
      key2 = (chr_a, chr_b, a, b)
 
      unambig = unambig_bins_get(key2, [])
 
      if a < bin_a:
        unambig = unambig[::-1]
 
      for pos_1, pos_2 in unambig:
        delta_1 = abs(pos_1-pos_a)
        delta_2 = abs(pos_2-pos_b)
 
        if delta_1 and delta_2 and (delta_1 < SEP_THRESHOLD) and (delta_2 < SEP_THRESHOLD):
          close.append((pos_1-pos_a, pos_2-pos_b))
 
        elif delta_1 > SEP_THRESHOLD and (a != bin_a):
          break
          
  if second_level:
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
 
    for key2, values in intermed_b.iteritems():
      for pos_2, delta_2 in values:
        for pos_1, delta_1 in intermed_a[key2]:
          delta_3 = abs(pos_1-pos_2)
 
          if delta_3 and delta_3 < SEP_THRESHOLD:
            close.append((delta_1, delta_2, delta_3))

  score_l = 1.0
  score_u = 1.0
  
  for deltas in close:
    s = 0.0
    
    for i, d in enumerate(deltas):

      if d < 0.0:
        d = max(-d, MIN_SEP)/SEP_THRESHOLD
        s = np.exp(-d*d/0.25)
        score_l += s
      else:
        d = max(d, MIN_SEP)/SEP_THRESHOLD
        s = np.exp(-d*d/0.25)
        score_u += s
    
  return np.log10(score_l * score_u)


def _write_ambig_filtered_ncc(in_file_path, out_ncc_path, resolved_ag=None, removed_ag=None):
  
  if not resolved_ag:
    resolved_ag = {}
    
  if not removed_ag:
    removed_ag = set()
  
  n = 0
  
  with open(out_ncc_path, 'w') as out_file_obj, open(in_file_path) as in_file_obj:
    write = out_file_obj.write

    for i, line in enumerate(in_file_obj):
      chr_a, start_a, end_a, f_start_a, f_end_a, strand_a, chr_b, start_b, end_b, \
        f_start_b, f_end_b, strand_b, ambig_group, pair_id, swap_pair = line.split()

      ambig_group = int(ambig_group)
      
      if ambig_group in resolved_ag:    
        if i in resolved_ag[ambig_group]:
          n == 1
          write(line)
         
      elif ambig_group not in removed_ag:  
        n += 1
        write(line)    
  
  msg = "Written {:,} of {:,} lines to {}".format(n, i+1, out_ncc_path)
  
  print(msg)
   
      
def _load_bin_sort_ncc(in_ncc_path):
  
  msg = 'Reading %s' % in_ncc_path
  print(msg)

  ag_data = defaultdict(list)
  all_contact_bins = defaultdict(list)
  
  with open(in_ncc_path) as in_file_obj:

    for i, line in enumerate(in_file_obj):
      chr_a, start_a, end_a, f_start_a, f_end_a, strand_a, chr_b, start_b, end_b, \
        f_start_b, f_end_b, strand_b, ambig_group, pair_id, swap_pair = line.split()

      ambig_group = int(ambig_group)

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
 
      bin_a = int(pos_a/SEP_THRESHOLD)
      bin_b = int(pos_b/SEP_THRESHOLD)
 
      key = (chr_a, chr_b, bin_a, bin_b)
      all_contact_bins[key].append((pos_a, pos_b, ambig_group))

      ag_data[ambig_group].append((key, pos_a, pos_b, i))

  msg = 'Loaded {:,} contact pairs in {:,} ambiguity groups'.format(i+1, len(ag_data))
  print(msg)
 
  msg = ' .. sorting data'
  print(msg)

  unambig_bins = {}
  chromo_pair_counts = defaultdict(int)
  chromo_bins = set()

  for key in all_contact_bins:
    chr_a, chr_b, bin_a, bin_b = key
    vals = sorted(all_contact_bins[key]) # Sort by 1st chromo pos
    unambig = [x[:2] for x in vals if len(ag_data[x[2]]) == 1]
 
    unambig_bins[key] = unambig
    all_contact_bins[key] = vals
 
    if unambig:
      chromo_bins.add((chr_a, bin_a))
      chromo_bins.add((chr_b, bin_b))
      
      chromo_pair_counts[(chr_a, chr_b)] += len(unambig)
      
  msg = ' .. done'
  print(msg)
 
  return ag_data, chromo_bins, all_contact_bins, unambig_bins, dict(chromo_pair_counts)


def remove_isolated_unambig(in_ncc_path, out_ncc_path, threshold=0.01):
  
  ag_data, chromo_bins, all_contact_bins, unambig_bins, chromo_pair_counts = _load_bin_sort_ncc(in_ncc_path)
  
  msg_template = ' .. Processed:{:>7,} Removed:{:>7,}'
  
  removed_ag = set()

  scores = []
  it = 0

  unambig_pairs = [(g, ag_data[g][0]) for g in ag_data if len(ag_data[g]) == 1]
 
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
      score = _get_network_score(chr_a, chr_b, pos_a, pos_b, bin_a, bin_b, unambig_bins, chromo_bins) 
    else:
      score = 0.0

    if score < threshold:
      removed_ag.add(ag)
  
  _write_ambig_filtered_ncc(in_ncc_path, out_ncc_path, resolved_ag=None, removed_ag=removed_ag)

  msg = msg_template.format(it, len(removed_ag))     
  print(msg)


def resolve_contacts(in_ncc_path, out_ncc_path, unambig_ncc_path=None, remove_isolated=False,
                     score_threshold=2.0, min_hc_relay=10.0):

  msg_template = ' .. Processed:{:>7,} Resolved:{:>7,}'

  msg = 'Reading contact data'
  print(msg)
  
  resolved_ag = {}
  removed_ag = set()
  
  ag_data, chromo_bins, all_contact_bins, unambig_bins, chromo_pair_counts = _load_bin_sort_ncc(in_ncc_path)

  if unambig_ncc_path:
    ag_data1, chromo_bins1, all_contact_bins1, unambig_bins, chromo_pair_counts1 = _load_bin_sort_ncc(unambig_ncc_path)  
 
  roots = set()
  chromos = set()
  for chr_a, chr_b in chromo_pair_counts:
    roots.add(chr_a.split('.')[0])
    roots.add(chr_b.split('.')[0])
    chromos.add(chr_a)
    chromos.add(chr_b)
  
  hc_close = {}  
  for root in roots:
    chr_a = root + '.a'
    chr_b = root + '.b' 
    n_inter = chromo_pair_counts.get((chr_a, chr_b), 0)
    
    for chr_c in chromos:
      if chr_c in (chr_a, chr_b):
        continue
      
      a = chromo_pair_counts.get((chr_a, chr_c), 0)
      b = chromo_pair_counts.get((chr_b, chr_c), 0)
      c = chromo_pair_counts.get((chr_c, chr_a), 0)
      d = chromo_pair_counts.get((chr_c, chr_b), 0)
      
      n_inter += np.sqrt((a + c) * (b + d))
      
    hc_close[(chr_a, chr_b)] = n_inter
    hc_close[(chr_b, chr_a)] = n_inter

  msg = 'Filtering with network scores'
  print(msg)

  scores = [[], [], [], []]
  it = 0  
    
  for ag in ag_data:
    it += 1
    if it % 1000 == 0:
      msg = msg_template.format(it, len(resolved_ag))
      print(msg)
 
    pairs = ag_data[ag]
 
    if len(pairs) == 1:
      chr_pair = tuple(pairs[0][0][:2])
 
      if chr_pair in hc_close and hc_close[chr_pair] < min_hc_relay:
        removed_ag.add(ag)
 
      continue

    line_indices = []
    group_scores = [0.0, 0.0, 0.0, 0.0]
    unambig_bins_get = unambig_bins.get
    
    for p, (key, pos_a, pos_b, line_idx) in enumerate(pairs):
      line_indices.append(line_idx)
      chr_a, chr_b, bin_a, bin_b = key
      contacts = all_contact_bins[key]
 
      if contacts:
        score = _get_network_score(chr_a, chr_b, pos_a, pos_b, bin_a, bin_b, unambig_bins, chromo_bins)
 
      else:
        score = 0.0
      
      group_scores[p] = score
     
    sort_idx = np.argsort(group_scores)[::-1]
    
    for i, j in enumerate(sort_idx):
      scores[i].append(group_scores[j])
    
       
    if group_scores[sort_idx[0]]  >  score_threshold * group_scores[sort_idx[1]]:
      resolved_ag[ag] = (line_indices[sort_idx[0]],)
    
    elif len(sort_idx) > 2 and group_scores[sort_idx[1]]  >  score_threshold * group_scores[sort_idx[2]]:
      resolved_ag[ag] = (line_indices[sort_idx[0]], line_indices[sort_idx[1]])
      
    elif group_scores[sort_idx[0]] == 0.0:
      if remove_isolated:
        removed_ag.add(ag)
    
  _write_ambig_filtered_ncc(in_ncc_path, out_ncc_path, resolved_ag, removed_ag)
  
  msg = msg_template.format(it, len(resolved_ag))
  print(msg)  


if __name__ == '__main__': 
  
  # All functions take an NCC file and create a filtered NCC file
  
  # Cleanup ambig so noise doesn't affect disambiguation too much
  
  remove_isolated_unambig('/data/hi-c/diploid/diploid_3_1CDS1-145.ncc', 'test_clean.ncc') 
  
  # The contact resolution iterative process 
  
  unambig_file = None # This can be set to use unambig from some file other than the main input
  
  # Permissive
  resolve_contacts('test_clean.ncc', 'test_filter_res1.ncc',
                   unambig_file, score_threshold=5.0)
  
  # More strict
  resolve_contacts('test_filter_res1.ncc', 'test_filter_res2.ncc', unambig_file)
  
  # Strict again and remove isolated ambigous
  resolve_contacts('test_filter_res2.ncc', 'diploid_3_1CDS1-145_resolved.ncc',
                   unambig_file, remove_isolated=True)


 
 
      
      
      
      
      
      
      
      
      
      
      
      
