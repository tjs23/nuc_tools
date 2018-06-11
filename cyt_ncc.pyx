from numpy cimport ndarray

  
def binContacts(ndarray[int, ndim=2] contacts,
                ndarray[int, ndim=2] binMatrix,
                int offsetA, int offsetB, int binSize=2000000,
                int symm=0, int transpose=0):
  
  cdef int i, a, b
  cdef int n, m, nCont = len(contacts[0])
  
  n = len(binMatrix)
  m = len(binMatrix[0])
  
  for i in range(nCont):
    a = (contacts[0,i]-offsetA)//binSize
    b = (contacts[1,i]-offsetB)//binSize
    if transpose:
      a, b = b, a
 
    if (0 <= a < n) and (0 <= b < m):
      binMatrix[a,b] += contacts[2,i]
 
      if symm and (a != b):
        binMatrix[b,a] += contacts[2,i]
  
  return binMatrix
