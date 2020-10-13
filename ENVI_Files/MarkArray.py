import numpy as np

def MarkArray3(n1,n2,n3=None) :
    if n3 == None :             # really a 2D array
        rv2 = np.zeros([n1,n2])
        for i in range(n1) :
            for j in range(n2) :
                rv2[i,j] = 10*i + j
        return rv2
    else :             # an actual 3D array
        rv3 = np.zeros([n1,n2,n3])
        for i in range(n1) :
            for j in range(n2) :
                for k in range(n3) :
                    rv3[i,j,k] = 100*i + 10*j + k
        return rv3

# Reverse actual index ordering of a from Fortran to C
def ToFortranOrder(a) :
    s = a.shape
    if 3 == len(s) :
        ra = np.zeros([s[2],s[1],s[0]], a.dtype)
        for i in range(s[0]) :
            for j in range(s[1]) :
                for k in range(s[2]) :
                    ra[k,j,i] = a[i,j,k]
        return ra
    elif 2 == len(s) :
        ra = np.zeros([s[1],s[0]], a.dtype)
        for i in range(s[0]) :
            for j in range(s[1]) :
                ra[j,i] = a[i,j]
        return ra
    elif 1 == len(s) :
        return a
    else :
        return None
