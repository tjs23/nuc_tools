import sys, os, time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict
from nuc_tools import util, io
from formats import bed, gff


"""
Extract intergenic and intronic regions for a GFF files exon and gene regions.
"""


def get_gff_chr_regions(gff_file, min_gene_len=1e2):
  
  data_dict = gff.load_data_track(gff_file, ['exon','gene'])
  intron_chr_regions = defaultdict(list)
  intergenic_chr_regions = defaultdict(list)
  long_gene_chr_regions  = defaultdict(list)
  short_first_exon_chr_regions = defaultdict(list)
  short_midd_exon_chr_regions = defaultdict(list)
  short_last_exon_chr_regions = defaultdict(list)
  short_gene_chr_regions = defaultdict(list)
  short_intron_chr_regions = defaultdict(list)
  first_exon_chr_regions = defaultdict(list)
  midd_exon_chr_regions = defaultdict(list)
  last_exon_chr_regions = defaultdict(list)
  
  chromos = sorted(data_dict['gene'])
  
  for chromo in chromos:    
    genes = data_dict['gene'][chromo]
    exons = data_dict['exon'][chromo]
    genes = genes[genes['pos2'].argsort()]
    exons = exons[exons['pos2'].argsort()]
    
    # Strand agnostic intergenic regions, between well-sized genes only
      
    gpos1 =  genes['pos1']
    pos2prev = -1
    len_prev = -1
    for i, pos2 in enumerate(genes['pos2']):
      pos1 = gpos1[i]
      len_curr = np.abs(pos2-pos1)
      
      if (len_prev > min_gene_len) and (len_curr > min_gene_len):
        intergenic_chr_regions[chromo].append((pos2prev, pos1, 1))
      
      len_prev = len_curr
      pos2prev = pos2
    
    intergenic_chr_regions[chromo] = np.array(intergenic_chr_regions[chromo], int)
    
    large_genes = genes[np.abs(genes['pos2'] - genes['pos1']) >= min_gene_len]
    long_gene_chr_regions[chromo] = np.stack([large_genes['pos1'], large_genes['pos2'], large_genes['strand']], axis=1)

    #small_genes = genes[np.abs(genes['pos2'] - genes['pos1']) < min_gene_len]
    #short_gene_chr_regions[chromo] = np.stack([small_genes['pos1'], small_genes['pos2'], small_genes['strand']], axis=1)
    
    pos_genes = genes[genes['strand'] == 1]
    pos_exons = exons[exons['strand'] == 1]
    
    neg_genes = genes[genes['strand'] == 0]
    neg_exons = exons[exons['strand'] == 0]
   
    epos2 =  pos_exons['pos2']
    epos1 =  pos_exons['pos1']
    
    eneg2 =  neg_exons['pos2']
    eneg1 =  neg_exons['pos1']

    gpos1 =  pos_genes['pos1']
    gneg1 =  neg_genes['pos1']

    gpos2 =  pos_genes['pos2']
    gneg2 =  neg_genes['pos2']
    
    pos_exon_gene_idx = np.searchsorted(pos_genes['pos2'], epos2)
    neg_exon_gene_idx = np.searchsorted(neg_genes['pos2'], eneg2)
    
    # + Strand

    pos2prev = -1
    j_prev = -1
    for i, pos2 in enumerate(epos2):
      pos1 = epos1[i]
      j = pos_exon_gene_idx[i] # Index of gene that contains the exon
      
      if j >= len(gpos1):
        break
      
      gene_start = gpos1[j]
      gene_end = gpos2[j]
      
      if (pos1 >= gene_start) and (pos1 < gene_end):
        if abs(gene_end-gene_start) < min_gene_len:
          if j == j_prev:
            short_intron_chr_regions[chromo].append((pos2prev, pos1, 1))
            
            if (i+1) < len(pos_exon_gene_idx) and pos_exon_gene_idx[i+1] == j: # Next exon in same gene
              short_midd_exon_chr_regions[chromo].append((pos1, pos2, 1))
            else:
              short_last_exon_chr_regions[chromo].append((pos1, pos2, 1))
            
          else: # First
            short_first_exon_chr_regions[chromo].append((pos1, pos2, 1))
            short_gene_chr_regions[chromo].append((gene_start, gene_end, 1))
       
        else:
          if j == j_prev:
            intron_chr_regions[chromo].append((pos2prev, pos1, 1))
            if (i+1) < len(pos_exon_gene_idx) and pos_exon_gene_idx[i+1] == j: # Next exon in same gene
              midd_exon_chr_regions[chromo].append((pos1, pos2, 1))
            else:
              last_exon_chr_regions[chromo].append((pos1, pos2, 1))
                  
          else:
            first_exon_chr_regions[chromo].append((pos1, pos2, 1))
        
      j_prev = j
      pos2prev = pos2
    
    
    # - strand
    
    pos2prev = -1
    j_prev = -1
    for i, pos2 in enumerate(eneg2):
      pos1 = eneg1[i]
      j = neg_exon_gene_idx[i]
      
      if j >= len(gneg1):
        break
      
      
      gene_start = gneg1[j]
      gene_end = gneg2[j]
      
      if (pos1 >= gene_start) and (pos1 < gene_end):
        if abs(gene_end-gene_start) < min_gene_len:
          short_intron_chr_regions[chromo].append((pos2prev, pos1, 0))
          
          if j == j_prev:
            if (i+1) < len(pos_exon_gene_idx) and pos_exon_gene_idx[i+1] == j: # Next exon in same gene
              short_midd_exon_chr_regions[chromo].append((pos1, pos2, 1))
            else:
              short_last_exon_chr_regions[chromo].append((pos1, pos2, 1))
          
          else: # First
            short_first_exon_chr_regions[chromo].append((pos1, pos2, 0))
            short_gene_chr_regions[chromo].append((gene_start, gene_end, 0))
        
        else:
          if j == j_prev:
            intron_chr_regions[chromo].append((pos2prev, pos1, 0))
            if (i+1) < len(pos_exon_gene_idx) and pos_exon_gene_idx[i+1] == j: # Next exon in same gene
              midd_exon_chr_regions[chromo].append((pos1, pos2, 0))
            else:
              last_exon_chr_regions[chromo].append((pos1, pos2, 0))
              
          else:
            first_exon_chr_regions[chromo].append((pos1, pos2, 0))
      
      j_prev = j
      pos2prev = pos2

    intron_chr_regions[chromo]     = np.array(intron_chr_regions[chromo], int)
    short_gene_chr_regions[chromo] = np.array(short_gene_chr_regions[chromo], int)
    short_intron_chr_regions[chromo] = np.array(short_intron_chr_regions[chromo], int)
    short_first_exon_chr_regions[chromo] = np.array(short_first_exon_chr_regions[chromo], int)
    short_midd_exon_chr_regions[chromo] = np.array(short_midd_exon_chr_regions[chromo], int)
    short_last_exon_chr_regions[chromo] = np.array(short_last_exon_chr_regions[chromo], int)
    first_exon_chr_regions[chromo] = np.array(first_exon_chr_regions[chromo], int)
    midd_exon_chr_regions[chromo] = np.array(midd_exon_chr_regions[chromo], int)
    last_exon_chr_regions[chromo] = np.array(last_exon_chr_regions[chromo], int)
  
  chr_region_dicts = [intron_chr_regions, first_exon_chr_regions,
                      midd_exon_chr_regions, last_exon_chr_regions,
                      intergenic_chr_regions, long_gene_chr_regions,
                      short_gene_chr_regions, short_intron_chr_regions,
                      short_first_exon_chr_regions,  short_midd_exon_chr_regions, short_last_exon_chr_regions]
  
  return chr_region_dicts

def write_bed_regions(bed_file_path, chr_region_dict):

  data_dict = {}
  for chromo in chr_region_dict:
    regions = chr_region_dict[chromo]
    
    if len(regions):
      value_anno = np.ones(regions.shape)
      value_anno[:,2] = 0
 
      data_dict[chromo] = np.concatenate([regions, value_anno], axis=1)

  bed.save_data_track(bed_file_path, data_dict)


def gff_to_further_bed_regions(out_file_root, gff_file, min_gene_len=10e3):
  
  chr_region_dicts = get_gff_chr_regions(gff_file, min_gene_len)
  
  tags = ('_introns.bed','_first-exon.bed',
          '_mid-exon.bed','_last-exon.bed',
          '_intergenic.bed','_long-gene.bed',
          '_short-gene.bed','_short-intron.bed',
          '_short-first-exon.bed','_short-mid-exon.bed','_short-last-exon.bed',)
  
  for chr_region_dict, tag in zip(chr_region_dicts, tags):
    write_bed_regions(out_file_root + tag, chr_region_dict)
  

if __name__ == '__main__':
  
   gff_path = '/data/dino_hi-c/hem_flye_4_ssRNA_Trinity.gff3'

   gff_to_further_bed_regions('/data/dino_hi-c/hem_flye_4_ssRNA_Trinity', gff_path)
  
