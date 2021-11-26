import sys, os, re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nuc_tools import io, util

from formats import bed, sam

sam_paths = sys.argv[1:]

size_ranges = [(0, 200), (200, 400), (400, 500), (500, 1000)]

for bin_size in (1000, 100):
  msg = 'Bin size: {}'.format(bin_size)
  
  print(msg)
  
  for size_range in size_ranges:
    a, b = size_range
    msg = '  Mol size range: {}-{}'.format(a, b)

    if bin_size < 1e3:
      file_end = '_sz{}-{}_{}bp.bed'.format(a, b, int(bin_size))
    else:
      file_end = '_sz{}-{}_{}k.bed'.format(a, b, int(bin_size/1e3))

    for sam_path in sam_paths:
 
      bed_path = os.path.splitext(sam_path)[0] + file_end
 
      util.info('Loading and binning {}'.format(sam_path))
 
      data_dict = sam.load_data_track(sam_path, bin_size, min_qual=10, num_cpu=8, mol_size_range=size_range)
 
      #for chromo in data_dict:
      #  data_dict[chromo]['strand'] = True
 
      util.info('Saving {}'.format(bed_path))
 
      bed.save_data_track(bed_path, data_dict, as_float=True)
 
      
    

