
loop_file = 'ES_loops.txt'

loop_bed_file = 'ES_loops_mm9.bed'

line_fmt = '%s\t%s\t%s\t%d\t%d\t+\n'

with open(loop_file) as file_obj, open(loop_bed_file, 'w') as out_file_obj:
  write = out_file_obj.write
  file_obj.readline()
  pid = 0
  
  for line in file_obj:
    chr1, s1, e1, chr2, s2, e2, color, obs = line.split()[:8]
    pid += 1
    
    obs = int(float(obs))
    line1 = line_fmt % (chr1, s1, e1, pid, obs)
    line2 = line_fmt % (chr2, s2, e2, pid, obs)
    write(line1)
    write(line2)
