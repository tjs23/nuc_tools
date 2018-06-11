from libc.math cimport abs, sqrt, ceil, floor, log, log2, acos, cos
from numpy cimport ndarray
import numpy as np
from numpy import ones, zeros, int32, float32, uint8, fromstring
from numpy import sort, empty, array, arange, concatenate, searchsorted

def pairRegionsIntersection(ndarray[int, ndim=2] pairs,
                            ndarray[int, ndim=2] regions,
                            exclude=False, allow_partial=False,
                            region_indices=False):
  
  cdef int i, j, k, a, b
  cdef int exc = int(exclude)
  cdef int partial = int(allow_partial)
  cdef int ni = 0
  cdef int np = len(pairs)
  cdef int nr = len(regions)
  cdef ndarray[int, ndim=1] indices = empty(np, int32)
  cdef ndarray[int, ndim=1] indices_reg = empty(np, int32)
  cdef ndarray[int, ndim=1] order = array(regions[:,0].argsort(), int32)  
  
  for i in range(np):
    
    if pairs[i,1] < regions[order[0],0]:
      if exc:
        indices[ni] = i
        ni += 1
      
      continue

    if pairs[i,0] < regions[order[0],0]:
      if exc and partial:
        indices[ni] = i
        ni += 1
      
      continue
      
    a = 0
    b = 0
    
    for k in range(nr):
      j = order[k]
      #print i, j, k
      
      if (regions[j,0] <= pairs[i,0]) and (pairs[i,0] <= regions[j,1]):
        a = 1
    
      if (regions[j,0] <= pairs[i,1]) and (pairs[i,1] <= regions[j,1]):
        b = 1
 
      if (pairs[i, 0] < regions[j, 0]) and (pairs[i, 1] < regions[j, 0]):
        break
      
      if partial & (a | b):
        break
      elif a & b:
        break
    
    if partial:
      if exc and not (a & b):
        indices[ni] = i
        indices_reg[ni] = j
        ni += 1
      
      elif a | b:
        indices[ni] = i
        indices_reg[ni] = j
        ni += 1
      
    else:
      if exc and not (a | b):
        indices[ni] = i
        indices_reg[ni] = j
        ni += 1
      
      elif a & b:
        indices[ni] = i
        indices_reg[ni] = j
        ni += 1

  
  if region_indices:
    return indices[:ni], indices_reg[:ni]
    
    
  else:
    return indices[:ni]

def regionBinValues(ndarray[int, ndim=2] regions, ndarray[double, ndim=1] values,
                    int binSize=1000, int start=0, int end=-1, double dataMax=0.0,
                    double scale=1.0, double threshold=0.0):
                    
  cdef int i, p1, p2, b1, b2, b3, s, e
  cdef int nBins, n = len(values)
  cdef double f, r, v, vMin, vMax
  
  if len(regions) != n:
    data = (len(regions), n)
    raise Exception('Number of regions (%d) does not match number of values (%d)' % data) 
  
  if end < 0:
    end = binSize * int32(regions.max() / binSize)
  
  s = start//binSize
  e = end//binSize
  nBins = 1+e-s
  
  cdef ndarray[double, ndim=1] hist = zeros(nBins, float)
  
  for i in range(n):
    
    v = values[i]
    if abs(v) < threshold:
      continue
    
    if regions[i,0] > regions[i,1]:
      p1 = regions[i,1] 
      p2 = regions[i,0]
    
    else:
      p1 = regions[i,0]
      p2 = regions[i,1]
    
    if end < p1:
      continue
    
    if start > p2:
      continue  
    
    b1 = p1 // binSize
    b2 = p2 // binSize
    
    if b1 == b2:
      if b1 < s:
        continue
      
      if b1 > e:
        continue
        
      hist[b1-s] += v

    else:
      r = <double> (p2-p1)
    
      for b3 in range(b1, b2+1):
        if b3 < s:
          continue
        
        if b3 >= e:
          break  
      
        if b3 * binSize < p1:
          f = <double> ((b3+1)*binSize - p1) / r 
        
        elif (b3+1) * binSize > p2:
          f = <double> (p2 - b3*binSize) / r 
        
        else:
          f = 1.0
      
        hist[b3-s] += v * f
  
  if dataMax != 0.0:
    vMin = hist[0]
    vMax = hist[0]
    
    for i in range(1, nBins):
      if hist[i] < vMin:
        vMin = hist[i]
      
      elif hist[i] > vMax:
        vMax = hist[i]
    
    vMax = max(abs(vMin), vMax, dataMax)

    if vMax > 0.0:
      for i in range(0, nBins):
        hist[i] = hist[i]/vMax
  
  for i in range(0, nBins):
    hist[i] = hist[i] * scale  
  
  return hist
  
def getInvDistSums(ndarray[double, ndim=2] coords_a not None,
                   ndarray[double, ndim=2] coords_b not None,
                   ndarray[long, ndim=1] seq_pos_a=None,
                   ndarray[long, ndim=1] seq_pos_b=None,
                   py_min_seq_sep=None, 
                   ndarray[double, ndim=1] values_b=None,
                   power_adj=1):

  cdef int i, j
  cdef double dx, dy, dz, d
  cdef int p = power_adj
  cdef int na = len(coords_a)
  cdef int nb = len(coords_b)
  cdef long min_seq_sep
  
  cdef ndarray[double, ndim=1] inv_dist_sums = zeros(na, float)
  
  if values_b is None:
    values_b = ones(nb, float)
  
  if py_min_seq_sep is None:
    for i in range(na):
      for j in range(nb):
        dx = coords_a[i, 0] - coords_b[j, 0]
        dy = coords_a[i, 1] - coords_b[j, 1]
        dz = coords_a[i, 2] - coords_b[j, 2]
        d = dx*dx + dy*dy + dz*dz
        
        if p > 1:
          d = d ** p
        
        if d > 0:
          inv_dist_sums[i] += values_b[j]/d
  
  else:
    if seq_pos_a is None or seq_pos_b is None:
      raise ValueError('py_min_seq_sep is not None, seq_pos_a and '
                       'seq_pos_b may not be None')
    min_seq_sep = py_min_seq_sep
  
    for i in range(na):
      for j in range(nb):
        if abs(seq_pos_a[i]-seq_pos_b[j]) < min_seq_sep:
          continue
        
        dx = coords_a[i, 0] - coords_b[j, 0]
        dy = coords_a[i, 1] - coords_b[j, 1]
        dz = coords_a[i, 2] - coords_b[j, 2]
        d = dx*dx + dy*dy + dz*dz
        
        if p > 1:
          d = d ** p
 
        if d > 0:
          inv_dist_sums[i] += values_b[j]/d  
  
  return inv_dist_sums

