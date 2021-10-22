import sys, os
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nuc_tools import util, io, formats


"""
For EDL

Pictures of typical cells at three time point (0, 24, 48 hr) with three different gene classes, highlighted vs trans distance
[9 images]

Three gene sets

Plot distribution structural values like trans distance, A depth, A/B boundary dist
 for the three gene sets 


tool to take a number of structures 


"""




struc_groups = ['/data/hi-c/hybrid/transition_state/Haploid/EDL_24h_Rex1High/*._10x_100kb.n3d',
                '/data/hi-c/hybrid/transition_state/Haploid/EDL_24h_Rex1Low /*._10x_100kb.n3d',
                '/data/hi-c/hybrid/transition_state/Haploid/EDL_48h/*._10x_100kb.n3d',
                '/data/hi-c/hybrid/transition_state/Haploid/EDL_ES_2iLIF/*._10x_100kb.n3d']


for group in struc_groups:
  
  struc_files = glob(group)
  
  for struc_file in struc_files:
    seq_pos_dict, coords_dict = formats.n3d.load_n3d_coords(struc_file)
  
    
