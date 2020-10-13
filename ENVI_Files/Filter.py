import numpy as np
import scipy.signal as ss
import scipy.ndimage as snd

def Gaussian(sigma, NFold=3) :
    """ Create a Gaussian kernal with width specified by signa.
    If extends NFold sigma in either direction (default 3)."""
    N = 1 + 2*NFold*sigma
    C = NFold*sigma
    t = (np.arange(0,N) - C)/sigma
    f0 = np.exp(-t**2)
    f = f0/f0.sum()
    return f

def Gaussian2D(sigma, sigmaY= -1, NFold=3) :
    """ Create a 2DGaussian kernal with width specified by signa.
    If sigmaY == -1 (the default) the use sigma in both directions.
    If extends NFold sigma in either direction (default 3)."""
    gX = Gaussian(sigma, NFold)
    if sigmaY <= 0 :
        gY = gX
    else :
        gY = Gaussian(sigmaY, NFold)
    return np.outer( gX, gY)
        
def Filter( Sequence, Kernel) :
    NPad = (Kernel.shape[0] - 1)//2 # force integer division
    SeqLen = Sequence.shape[0]
    SeqTmp = np.ones((SeqLen + 2*NPad,))
    LeadVal = np.mean(Sequence[:5])
    TrailVal = np.mean(Sequence[-5:])
    SeqTmp[:NPad] = LeadVal
    SeqTmp[(-NPad):] = TrailVal
    SeqTmp[NPad:(NPad+SeqLen)] = Sequence
    # print(SeqTmp)
    fs0 = np.convolve(SeqTmp, Kernel,"same")
    fs = fs0[NPad:(NPad+SeqLen)]
    return fs # , fs0, SeqTmp


# Cheerfully stolen from https://dsp.stackexchange.com/questions/41184/high-pass-filter-in-python-scipy
def highpass_filter(y, sr, filter_stop_freq = 0.1, filter_pass_freq = 0.2, filter_order = 101):
    """ Apply a high pass filter with specified stop, pass and #taps specified to  a 1-D
    sequence y.  sr is the sampling frequency.
    Returns the filtered sequence.
    """
    # In the original posting, for the audio case
    # filter_stop_freq = 70  # Hz
    # filter_pass_freq = 100  # Hz
    # filter_order = 1001
    
    # High-pass filter
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = ss.firls(filter_order, bands, desired, nyq=nyquist_rate)
    
    # Apply high-pass filter
    filtered_signal = ss.filtfilt(filter_coefs, [1], y)
    return filtered_signal

def Monotone(N, Delta, A, Period, Phase=0) :
    """Create a single frequency signal signal
    N : Number of data points, Delta : Sampling frequency
    A : Amplitude, Period : signal period (1/f) """
    Mult = 2.0 * np.pi * Delta/Period
    s = A * np.sin(Mult*np.arange(N) + Phase)
    return s

def TestFilter1(N, Signal, Filter) :
    Delta, A, Period = Signal   # unpack it
    ss = Monotone(N, Delta, A, Period) # test signal
    # print(N, Delta, A, Period)
    # Filter the signal
    sr = 0.5/Delta             # Nyquist frequency
    filter_stop_freq, filter_pass_freq , filter_order =  Filter # Unpack this one too
    # print(sr, filter_stop_freq, filter_pass_freq, filter_order)
    sf = highpass_filter(ss, sr, filter_stop_freq, filter_pass_freq, filter_order)
    return (np.sum(ss**2), np.sum(sf**2))

def TestFilter(Freq, N, Signal, Filter, FreqIsPeriod=False) :
    Delta, A, Period = Signal   # unpack it
    IsList = False
    if isinstance(Freq, list) or isinstance(Freq, tuple) : # indexed the same way
        RetVal = []
        IsList = True
    elif isinstance(Freq, np.ndarray) :
        RetVal = np.zeros([Freq.shape[0], 4])
    else :
        return None
    nn = 0
    for f in Freq :             # replace the period with frequency  equivalent
        if not FreqIsPeriod :
            f = 1.0/f           # now it's a period
        # print(f)
        SignalF = (Delta, A, f)
        sf = TestFilter1(N, SignalF, Filter)
        ThisVal = (f, sf[0], sf[1], sf[1]/sf[0])
        if IsList :
            RetVal.append(ThisVal)
        else :
            RetVal[nn,:] = ThisVal
            nn +=1
    return RetVal

def TestFilterSet(N, Signal, Filter) :
    """ Provide a place to set the frequencies"""
    # Freq = np.array([200., 100., 74.98942094242997, 56.23413252026576, 42.16965033684151, 31.62277660336759,
    #                  23.71373705822385, 17.78279410008404, 13.3352143222104, 10., 7.498942094242997, 5.623413252026576,
    #                  4.216965033684151, 3.162277660336759, 2.371373705822385, 1.778279410008404, 1.33352143222104,
    #                 1.0, 0.7498942094242997, 0.5623413252026576, 0.4216965033684151, 0.3162277660336759, 0.2371373705822385, 0.2])
    # RetVal = TestFilter(Freq, N, Signal, Filter, False)
    Period = np.array([ 0.2,  0.3,  0.4,  0.6,  0.8,  1.2,  1.6,  1.8,  2. ,  2.2,  2.4,
                        2.6,  2.8,  3.2,  3.8,  4.8,  6.4,  7.5, 8.5, 9.6, 10.6, 11.7, 12.8, 19.2, 25.6])
    RetVal = TestFilter(Period, N, Signal, Filter, True)
    return RetVal

def TestFilterSets(N, Signal, Filters) :
    """Map out a series of filters.  Other things are comperable, so the frequencies should be the same."""
    if isinstance(Filters, list) or isinstance(Filters, tuple) :
        Filters = np.array(Filters)
    if isinstance( Filters, np.ndarray) : # should always be true
        RetVal = np.zeros([25, Filters.shape[0]]) # generate a spectrum for each filter
    else :
        return None

    for nn in range(Filters.shape[0])  :
        rv = TestFilterSet(N, Signal, Filters[nn, :])
        RetVal[:,nn] = rv[:,3]

    return (rv[:,0], RetVal)

def FilterImage( H, K, MakeCopy=False) :
    """ Apply to 2D kernel K to each image (slice) in the hypercube H
    and return the filtered hypercube."""
    rv = H                      # note that this is just a reference to the input
    if MakeCopy :
        rv = np.copy(H)
    # Filter in place whatever we're going to hand back
    rank = len(H.shape)         # umber of dimensions
    if rank == 2 :              # A single image slice - useful for testing
        snd.convolve( rv , K, output=rv )
    else :
        N = H.shape[2]              # Number of slices
        for i in range( N ) :
            snd.convolve( rv[:,:,i], K, output=rv[:,:,i])
    return rv


