import sys, os
from glob import glob

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('/home/tjs23/ihs/')

from NewBlast import BlastSearch

blastdb_path_a = '/data/genome/mm_129_db'
blastdb_path_b = '/data/genome/mm_CAST_db'

fastq_path_1 = 'test_HybridES0418_002_reads1_clipped__unmapped.fq.gz'
fastq_path_2 = 'test_HybridES0418_002_reads2_clipped__unmapped.fq.gz'

from nuc_tools import util, io


with io.open_file(fastq_path_1) as file_obj_1, io.open_file(fastq_path_2) as file_obj_2:
  readline1 = file_obj_1.readline
  readline2 = file_obj_2.readline
  n_reads = 0

  lineA1 = readline1()
  lineA2 = readline2()
  while lineA1 and lineA2:
    n_reads += 1
    
    seq1 = readline1()[:-1]
    seq2 = readline2()[:-1]
    
    lineC1 = readline1()
    lineC2 = readline2()
    
    lineD1 = readline1()
    lineD2 = readline2()
    
    lineA1 = readline1()
    lineA2 = readline2()
    
    for label, seq, db in (('1A', seq1, blastdb_path_a),
                           ('1B', seq1, blastdb_path_b),
                           ('2A', seq2, blastdb_path_a),
                           ('2B', seq2, blastdb_path_b)):
    
      print seq
          
      result = BlastSearch(seq, db, blastExe='blastn', eCutoff=10, cpuCores=None).run()
 
      for i, hit in enumerate(result.hits[:3]):
        print '%s:%d:%d' % (label,i,n_reads)
        if result:
          print '  Q:{:,}-{:,} M:{:,}-{:,}'.format(hit.queryFrom, hit.queryTo, hit.hitFrom, hit.hitTo)
          print hit.qseq
          print hit.midline
          print hit.hseq
        else:
          print "?"
      
      print
