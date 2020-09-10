# Asymetric least squares smoothing for baseline correction (Eiler & Boelens method)
# Cheerfully lifted from https://stackoverflow.com/questions/29156532/python-baseline-correction-library/29185844

import scipy.sparse
import scipy.sparse.linalg
import numpy as np

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = scipy.sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = scipy.sparse.linalg.spsolve(Z, w*y)
    wOld = w                    # be able to look for changes
    w = p * (y > z) + (1-p) * (y < z)
    if np.all( wOld == w) :
        # print("Break at %d" % (i,))
        break                   # no change means we've converged
  return z

def baseline_als2(y, lam, p, niter=10):                                                                        

    s  = len(y)                                                                                               
    # assemble difference matrix                                                                              
    D0 = scipy.sparse.eye( s )                                                                                      
    d1 = [numpy.ones( s-1 ) * -2]                                                                             
    D1 = scipy.sparse.diags( d1, [-1] )                                                                             
    d2 = [ numpy.ones( s-2 ) * 1]                                                                             
    D2 = scipy.sparse.diags( d2, [-2] )                                                                             

    D  = D0 + D2 + D1                                                                                         
    w  = np.ones( s )                                                                                         
    for i in range( niter ):                                                                                  
        W = scipy.sparse.diags( [w], [0] )                                                                          
        Z =  W + lam*D.dot( D.transpose() )                                                                   
        z = scipy.sparse.linalg.spsolve( Z, w*y )
        wOld = w                    # be able to look for changes
        w = p * (y > z) + (1-p) * (y < z)                                                                     
        if np.all( wOld == w) :
            # print("Break at %d" % (i,))
            break                   # no change means we've converged

    return z

