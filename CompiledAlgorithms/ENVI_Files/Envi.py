# Read the image block in ENVI format
# Only handles the things ChemVision writes (not all the items ENVI can write)
import re
import os
import numpy as np
import scipy.signal as ss
import imageio
from CompiledAlgorithms.ENVI_Files import MarkArray, Alss, Filter

class EnviImage :
    def __init__(self) :
        self.description = ""
        self.samples = -1
        self.lines = -1
        self.bands = -1
        self.header_offset = 0
        self.file_type = "Spero Chemvision"
        self.data_type = -1
        self.interleave = None
        self.sensor_type = None
        self.byte_order = 0
        self.wavelength_units = "cm-1"
        self.wavelength = []
        self.Pixels = []
        self.PixelsArchive = [] # list of hypercubes after Pixels is filled with a processed version
        self.StoredAsTransmittance = False

    def __str__(self) :
        s = "ENVI" + os.linesep
        s += "description = %s" % (self.description,) + os.linesep
        s += "samples = %d" % (self.samples,) + os.linesep
        s += "lines = %d" % (self.lines,) + os.linesep
        s += "bands = %d" % (self.bands,) + os.linesep
        s += "header offset = %d" % (self.header_offset,) + os.linesep
        s += "file type = %s" % (self.file_type,) + os.linesep
        s += "data type = %d" % (self.data_type,) + os.linesep
        s += "interleave = %s" % (self.interleave,) + os.linesep
        s += "sensor type = %s" % (self.sensor_type,) + os.linesep
        s += "byte order = %d" % (self.byte_order,) + os.linesep
        s += "wavelength units = %s" % (self.wavelength_units,) + os.linesep
        s += "wavelength = { "
        if 0 < len(self.wavelength) :
            for w in self.wavelength[:-1] :
                s += " %.1f," % (w,)
            s += " %.1f" % (self.wavelength[-1],) # last one has no trailing comma
        s += "}" + os.linesep
        # Don't include the pixels in the string representation
        return s

    def AsMHD(self, VoxelFile=None, SingleSlice=False) :
        # Represent the header information in meta header format        
        s = "ObjectType = Image" + os.linesep
        s += "BinaryData = True" + os.linesep
        s += "CompressedData = False" + os.linesep
        s += "NDims = "
        if SingleSlice :
            s += "2" + os.linesep
            s += "TransformMatrix = 1 0 0 1" + os.linesep
            s += "Offset = 0 0" + os.linesep
            s += "CenterOfRotation = 0 0" + os.linesep
            s += ("DimSize = %d %d" % (self.samples, self.lines)) + os.linesep
        else:
            s += "3" + os.linesep
            s += "TransformMatrix = 1 0 0 0 1 0 0 0 1" + os.linesep
            s += "Offset = 0 0 0" + os.linesep
            s += "CenterOfRotation = 0 0 0" + os.linesep
            s += ("DimSize = %d %d %d" % (self.samples, self.lines, self.bands)) + os.linesep
        s += "ElementType = MET_FLOAT" + os.linesep
        if VoxelFile != None :
            s += "ElementDataFile = " + VoxelFile  + os.linesep
        # Don't include the pixels in this representation
        return s
        
    def Ascii(self, FileType=None) :
        if FileType == None or FileType == 'hdr' :
            return self.__str__()
        elif  FileType == 'mhd' :
            pass

    def SubSample(self, PixelSlice=False, ChangeLinesSamples=False) :
        """ Replace members of the image by their subsampled equivalents.
        """
        if not PixelSlice :     # no-op
            return None
        if type([]) == type(self.Pixels) :
            # Pixels have not yet been set
            if not ChangeLinesSamples :
                return None
            # Adjust samples and lines (if set) even though there are no pixels
            if self.samples == -1 or self.lines == -1 :
                return None     # those haven't been set either
            if PixelSlice == True :
                print("SubSample - autoquarter- : ", self.samples, self.lines)
                self.samples = int(self.samples/2)
                self.lines   = int(self.lines/2)
                print("SubSample - autoquarter+ : ", self.samples, self.lines)
                return True
            return None         # not yet supported
        
        if PixelSlice == True : # take middle of right half of image
            Size = self.Pixels.shape
            LowRow = int(Size[0]/4);    HighRow = 3*LowRow
            LowCol = int(Size[1]/2);	HighCol = Size[1]
        elif  type((1,2,3)) == type(PixelSlice) or type([1,2,3]) == type(PixelSlice) :
            # For the moment we require all four items to be spectified
            if len(PixelSlice) != 4 :
                return None     # invalid input
            LowRow, HighRow, LowCol, HighCol = PixelSlice
        # Get the desired subsample
        SubPix = self.Pixels[LowRow:HighRow, LowCol:HighCol]
        self.samples = HighRow - LowRow
        self.lines = HighCol - LowCol
        self.Pixels = SubPix
        return True

    def Clone(self) :
        """Return a deep copy of self."""
        eiClone = EnviImage()   # make a new copy to load up
        eiClone.description = self.description
        eiClone.samples = self.samples
        eiClone.lines = self.lines
        eiClone.bands = self.bands
        eiClone.header = self.header_offset
        eiClone.file = self.file_type
        eiClone.data = self.data_type
        eiClone.interleave = self.interleave
        eiClone.sensor = self.sensor_type
        eiClone.byte = self.byte_order
        eiClone.wavelength = self.wavelength_units
        eiClone.StoredAsTransmittance = self.StoredAsTransmittance
        # Only need to do a deep copy of things which are typically containers
        eiClone.wavelength = self.wavelength
        eiClone.Pixels = np.copy(self.Pixels)
        eiClone.PixelsArchive = [] # Don't carry original object's archive along
        return eiClone

    def Conforms(self, Denom) :
        """ Do self and Denom match?  I.e., have the same shape and wavelengths.
        Note that the Pixels have to be the same shape, but not the same data type,
        values or even absorption/transmission state.
        """
        # First, check if it's an array.  If so we can see if they have the same shape
        # Note that if one of them is an array the other one also must be in order for
        # them to conform.        
        if isinstance(self.Pixels, np.ndarray) :
            if isinstance(Denom.Pixels, np.ndarray) :
                if self.Pixels.shape != Denom.Pixels.shape :
                    return False
            else :
                return False    # self is an array and Denom is not --> mismatch
        elif  isinstance(Denom.Pixels, np.ndarray) :
            return False        # Denom is an array and self is not --> mismatch
        # Handle non-array case
        elif self.Pixels != Denom.Pixels : # Note that these can both be [] or None and this is OK
            return False
        # Move on to wavenumber
        if self.wavelength == [] and Denom.wavelength == [] : # this is the case produced by the default contructor
            return True
        if np.all(np.array(self.wavelength) == np.array(Denom.wavelength)) : # Force to array to make it work with lists
            return True
        return False

    def DecimateSpectrum( self, DecimationFactor ) :
        """ Decimate the wavenumber vector and spectral part of the image hypercube by DecimationFactor.
        DecimationFactor must be a snakk integer 1 <= DecimationFactor <= self.Pixels.shape[2]/2
        DecimationFactor == 1 is a no-op."""
        if not isinstance(DecimationFactor, int ) :
            return False     # This probably should raise an exception
        if DecimationFactor <= 0  : # Probably should raise and exception
            return False
        if DecimationFactor == 1 : # Already there
            return True
        NL = np.array(self.wavelength, copy=False).shape[-1] # this should be the number of wavenumbers
        if DecimationFactor > NL // 2 :
            return False        # Should be an error
        # All OK, do the deed
        self.Pixels = self.Pixels[:,:,::DecimationFactor]
        self.wavelength = self.wavelength[::DecimationFactor]
        return True

    def CreateRatioImage(self, Denom, ResultInAbsorption=True) :
        """Create a new image with self as the numerator and Denom as the denominator.
        Note that the images must conform, having the same shape and with equal wavelengths.
        """
        if not self.Conforms(Denom) :
            return None         # probably should raise an error instead
        Output = self.Clone()
        # Note that the division must be done in transmission, so we need to
        # make sure we do the division on that form of the data.
        if Output.StoredAsTransmittance :
            if Denom.StoredAsTransmittance :
                Output.Pixels = Output.Pixels / Denom.Pixels # numpy doesn't seem to have a /= operator
            else :                                           # Denom is in absorbance
                Output.Pixels = Output.Pixels / 10.0**(-Denom.Pixels) # use value converted to transmittance
        else :                  # The numerator (Output) is in absorbance
            if Denom.StoredAsTransmittance :
                Output.Pixels = 10.0**(-Output.Pixels) / Denom.Pixels
            else :                                           # They're both absorbance
                Output.Pixels = 10.0**(Denom.Pixels - Output.Pixels) # use value converted to transmittance
        # At this point Output.Pixels is in the form of a ratio of transmittance.  Do we want to change it?
        Output.StoredAsTransmittance = True # mark the current state
        if ResultInAbsorption :
            Output.Pixels = -np.log10(Output.Pixels)
            Output.StoredAsTransmittance = False # Flip it back
        return Output
    
    def Read(self, FileName, bReadImageData=True, bEchoKeys=False, bTransmittance=True):
        #HeaderFileName = FileName + ".hdr" # FileName is the name of the pixel file
        HeaderFileName = os.path.splitext(FileName)[0]+'.hdr'
        with open(HeaderFileName, "r") as f :
            # First line must be ENVI
            l = f.readline().strip()
            if l != "ENVI" :
                raise SyntaxError # this is cheating, since it's input file syntax that causes the problem
            for l in f.readlines() :
                ls = [t.strip() for t in l.split('=')]
                if len(ls) < 2 :
                    continue    # don't need to bother with empty lines
                Key, Entry = ls
                if bEchoKeys :
                    print("Key=<%s> Entry=<%s>" % (Key, Entry))
                if Key == "description" :
                    self.description = Entry
                elif Key == "samples" :
                    self.samples = int(Entry)
                elif Key == "lines" :
                    self.lines = int(Entry)
                elif Key == "bands" :
                    self.bands = int(Entry)
                elif Key == "header offset" :
                    self.header_offset = int(Entry)
                elif Key == "file type" :
                    self.file_type = Entry
                elif Key == "data type" :
                    self.data_type = int(Entry)
                elif Key == "interleave" :
                    self.interleave = Entry
                elif Key == "sensor type" :
                    self.sensor_type = Entry
                elif Key == "byte order" :
                    self.byte_order = int(Entry)
                elif Key == "wavelength units" :
                    self.wavelength_units = Entry
                elif Key == "wavelength" :
                    # Burst out all the wavelengths
                    self.wavelength = [float(x) for x in re.sub('[{},]', ' ', Entry).split()]
                else :
                    pass
        # Now slurp up the pixels
        if bReadImageData and self.data_type == 4 : # only 32 bit float is supported (and all CV writes)
            self.Pixels = np.fromfile( FileName,  np.dtype("float32"), -1, "")
            # Stored values are absorbance - do we want to flip them over?
            self.StoredAsTransmittance = bTransmittance
            if bTransmittance :
                self.Pixels = 10 ** -self.Pixels
            self.Pixels = self.Pixels.reshape(( self.samples, self.lines, self.bands),order='F')

    def ReadBackground(self, BaseFileName, toCounts=True ) :
        """ Read the pixels in from file BaseFileName, assumed to be in ENVI format (without the .hdr).
        Note that the pixels are always stored as decadic absorbance but probably should end up in
        transmission (since the really represent raw counts).
        If toCounts is True (the default) the resulting values will be converted back to counts,
        otherwise the pixels will be left as decadic absorbance (as storted).
        Lets the regular Read() function do the dirty work.
        """
        self.Read(BaseFileName, True, False, False) # read a absorbance and convert it ourselves
        if toCounts :
            self.Pixels = 1000.0 * np.log(10.0) * self.Pixels # undo the 'decadic absorbance' thing

    def ReadImage(self, Filename) :
        """ Read an image file (jpeg, png, &c ... whatever types imageio recognizes) into this format
        with sane (but not necessarily good) settings for the member fields.  In particular, the wavelength
        fields are sequential integers (without physical meaning).  Users are welcome to change these
        appropriately once the image has been read.
        Pixels are unually uint8, but they're converted here to float32 so they look like
        the regular images we read.  (Feel free to convert them back if that's convenient.
        """
        imTemp = imageio.imread(Filename)
        self.Pixels = np.array(imTemp, np.dtype('float32'))
        self.description = "Imported from image file " + Filename
        self.samples = imTemp.shape[0]
        self.lines = imTemp.shape[1]
        self.bands = imTemp.shape[2]
        self.header_offset = 0
        self.file_type = "Spero Chemvision"
        self.data_type = 4      # We forced things to be float32
        self.interleave = None
        self.sensor_type = None
        self.byte_order = 0
        self.wavelength_units = "Unknown"
        self.wavelength = list(range(self.bands))
        self.PixelsArchive = [] # list of hypercubes after Pixels is filled with a processed version
        self.StoredAsTransmittance = False
        return imTemp
        
    def Write(self, BaseFileName, SliceIndex=None ) :
        # If the file has no extension it will be saved as an ENVI format, base filename + hdr file
        # If the file name ends with '.hdr' it will again be saved as an ENVI file (strip hdr for voxel data)
        # If the file name ends with '.mhd' it will be saved as a MetaImage file (strip mhd for voxel data)
        # If SliceIndex is set to an integer that is a valid wavenumber index a single slice is saved in 2D mhd format
        # Start by sorting through possible file types (Handle to 2D mhd case separately)
        if type(3) == type(SliceIndex ) :
            if SliceIndex < 0 or self.Pixels.shape[2] <= SliceIndex :
                return          # index is out of range
            else :
                if '.mhd' == BaseFileName[-4:] : # Remove the extension from the voxel file name
                    HeaderFileName = BaseFileName
                    BaseFileName = BaseFileName[:-4]
                else :          # Add the extension to the header name
                    HeaderFileName = BaseFileName + ".mhd"
                OutputType = 'mhd1'
        else :
            if '.hdr' == BaseFileName[-4:] :
                HeaderFileName = BaseFileName
                BaseFileName = BaseFileName[:-4]
                OutputType = 'hdr'
            elif '.mhd' == BaseFileName[-4:] :
                HeaderFileName = BaseFileName
                BaseFileName = BaseFileName[:-4]
                OutputType = 'mhd'
            else :                                 # Assume this is the base name, create the header nuame
                HeaderFileName = BaseFileName + ".hdr" # FileName is the name of the pixel file
                OutputType = 'hdr'
        with open(HeaderFileName, "w") as f :
            if OutputType == 'hdr' :
                f.write(str(self))
            elif OutputType == 'mhd' :
                s = self.AsMHD(BaseFileName, False)
                f.write(s)
            elif OutputType == 'mhd1' :
                s = self.AsMHD(BaseFileName, True)
                f.write(s)
            else :
                return          # ?!?!?!
        if self.Pixels != [] :
            with open(BaseFileName, "wb") as f :
                if OutputType == 'mhd1' :
                    SlicePixels = MarkArray.ToFortranOrder(self.Pixels[:,:,SliceIndex])
                    # Do we need to transpose this?
                    if self.StoredAsTransmittance : # need to undo this - we assume stored stuff is absorbance
                        print("Converting 2D pixel values to absorbance")
                        SlicePixels = -np.log10(SlicePixels) # Convert it back
                        print("2D pixel values are " + str(SlicePixels.dtype))
                    SlicePixels.tofile(f)
                else :          # ENVI and 3D MHD both save the entire brick of voxels
                    # There's got to be a better way to do this than copying over the whole thing!!!
                    FortranPixels = MarkArray.ToFortranOrder(self.Pixels)
                    if self.StoredAsTransmittance : # need to undo this - we assume stored stuff is absorbance
                        print("Converting pixel values to absorbance")
                        FortranPixels = -np.log10(FortranPixels) # Convert it back
                    FortranPixels.tofile(f)
        return HeaderFileName

    def ApplyBaselineCorrection( self, ELambda, Ep, ArchiveOld=False, NIterations=10) :
        """Apply the baseline correction method of  Eilers and Boelens to the pixel data.
        Note that it really only works for absorption spectra, so convert to/from if data is transmission.
        If ArchiveOld is True save the previous pixels in case we want them back.
        The paper recommends 10^2 < ELambda < 10^9 and 0.001 < Ep < 0.1
        """
        if ArchiveOld :
            self.PixelsArchive.append(self.Pixels)
        if self.StoredAsTransmittance :
            PixelsA=-np.log10(self.Pixels)	# convert from transmittance to absorbance
        else :
            PixelsA= self.Pixels # Note that baseline correction doesn't change Pixels, so reference is OK
        Shape = PixelsA.shape
        PixelsAB = np.zeros((Shape[0]*Shape[1],Shape[2]), PixelsA.dtype)
        PixelsAR = PixelsA.reshape((Shape[0]*Shape[1],Shape[2]))
        print(PixelsAB.shape, PixelsAR.shape)
        # Compute the baseline and remove it
        for k in range(Shape[0]*Shape[1]) : 
            PixelsAB[k,:] = PixelsAR[k,:] - Alss.baseline_als(PixelsAR[k,:], ELambda, Ep, NIterations)
        # Make sure we leave the voxels in the right state and restore the original shape
        if self.StoredAsTransmittance :
            self.Pixels = np.reshape(10 ** -PixelsAB, Shape) # back to transmittance
        else :
            self.Pixels = np.reshape(PixelsAB, Shape)

    def ApplyMask( self, Mask, ReplaceValue=None, DoClose=False, ArchiveOld=False ):
        """ Replace Pixels with Pixels masked by Mask.  Mask must have the
        same shape as each image, so Mask.shape must equal Pixels.shape[0:2].
        Each element of Mask must be interpretable as a boolean value - keep the
        values where this is True.
        Pixels outside the mask are replaced with ReplaceValue.  If it is None (the default)
        it is set to the "nothing there" value.  If StoredAsTransmittance then 1 otherwise
        (absorbance) it is set to 0.  If ReplaceValue is an array it must have one dimension
        that is the same as the number of wavenumber (Pixels.shape[2]).
        """
        if ReplaceValue == None :
            ReplaceValue = np.ones(self.Pixels.shape[2]) if self.StoredAsTransmittance else np.zeros(self.Pixels.shape[2])
        elif len(ReplaceValue) != self.Pixels.shape[2] :
            raise IndexError    # not exactly, but close enough

        NumberOfSpectraMasked = 0
        for i in range(self.Pixels.shape[0]) :
            for j in range(self.Pixels.shape[1]) :
                if not Mask[i,j] :
                    self.Pixels[i,j,:] = ReplaceValue
                    NumberOfSpectraMasked += 1
        return NumberOfSpectraMasked
               
    def FilterAll( self, Kernel, ArchiveOld=False ):
        """Replace Pixels with Pixels spectrally filtered with Kernel.
        """
        if self.Pixels == [] :
            return None         # no pixels, nothing to do
        if ArchiveOld :
            self.PixelsArchive.append(self.Pixels)
        # Apply filtering to spectra belonging to each pixel
        PixelsC = np.zeros(self.Pixels.shape, self.Pixels.dtype)
        NN = 0
        for ii in range(self.Pixels.shape[0]) :
            for jj in range(self.Pixels.shape[1]) :
                PixelsC[ii,jj,:] = Filter.Filter(self.Pixels[ii,jj,:], Kernel)
                NN += 1
        # print(NN)
        self.Pixels = PixelsC
        return self.Pixels.shape[0]*self.Pixels.shape[1] # Number of spectra filtered

    def FilterAllHighPass( self, filter_stop_freq = 0.1, filter_pass_freq = 0.2, filter_order = 1001, ArchiveOld=False ):
        """Replace Pixels with Pixels spectrally filtered with a scipy.signal high pass filter of order filter_order.
        The transition goes from filter_stop_freq to filter_pass_freq
        """
        if self.Pixels == [] :
            return None         # no pixels, nothing to do
        if self.wavelength == [] : # required to find Nyquist frequency
            return None
        if ArchiveOld :
            self.PixelsArchive.append(self.Pixels)
        # Compute the filter once (and then apply it for each spectrum)
        # Find Nyquist frequency from wavelength (assume things are evenly spaced)
        nyquist_rate = 1.0/(2.0 * (self.wavelength[1] - self.wavelength[0]))
        print(nyquist_rate)
        desired = (0, 0, 1, 1)
        bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
        filter_coefs = ss.firls(filter_order, bands, desired, nyq=nyquist_rate)

        # Apply filtering to spectra belonging to each pixel
        PixelsC = np.zeros(self.Pixels.shape, self.Pixels.dtype)
        # NN = 0
        for ii in range(self.Pixels.shape[0]) :
            for jj in range(self.Pixels.shape[1]) :
                PixelsC[ii,jj,:] =  ss.filtfilt(filter_coefs, [1], self.Pixels[ii,jj,:])
                # NN += 1
        # print(NN)
        self.Pixels = PixelsC
        return self.Pixels.shape[0]*self.Pixels.shape[1] # Number of spectra filtered

    def SpectraROI(self, Regions) :
        """Find the spectra corresponding to each entry in Regions. 
        Regions must have the form [[LowRow, HighRow, LowCol, HighCol], ...]
        Error checking is marginal.
        Returns a series of spectra, one per column (since that works well with matplotlob.pyplot.plot() )
        """
        # Try to handle both lists and arrays
        NRegions = 0
        if isinstance(Regions, list) or isinstance(Regions, tuple) :
            NRegions = len(Regions)
            if NRegions == 0 or len(Regions[0]) != 4 : # only check one, hopefully OK
                return False    # probably should raise an error
            # print("List : NRegions = %d" % (NRegions))
        elif isinstance(Regions, np.ndarray) :
            NRegions = Regions.shape[0]
            if Regions.shape[1] != 4 or NRegions == 0 :
                return False    # probably should raise an error
            else :              # Don't know what sort of sequence this is
                return False    # probably should raise an error
            # print("Array : NRegions = %d" % (NRegions))

        # Check that we actually have pixels
        if not isinstance(self.Pixels, np.ndarray) :
            return False    # probably should raise an error
        Spectra = np.zeros([self.Pixels.shape[-1], NRegions], 'float64')
        # print( Spectra.shape )
        i = 0
        for Reg in Regions :
            # print( i, Reg )
            Spectra1 = self.Pixels[Reg[0]:Reg[1], Reg[2]:Reg[3], :]
            # print( Spectra1.shape )
            # Spectra[:,i] = np.mean(self.Pixels[Reg[0]:Reg[1], Reg[2]:Reg[3], :], (0, 1))
            Spectra[:,i] = np.mean( Spectra1, (0, 1))
            i += 1
        return Spectra

    def FillSlices(self, OnlyFindGaps=False) :
        """ Fill the wavenumber spacing so that there are no missing images.
        Assume that we step through consistancy from first to last by the step
        used from the first to second and interpolate the missing slices.
        (Note that this means that non-evenly spaced wavenumbers will result in garbage.)
        If OnlyFindGaps==True (default False) it just finds the wavenumbers where it
        would conjure images and returns them but does not change self.
        If it is True is interpolates the images in Pixel and corrects wavenumber.
        For the moment it only handles a single gap (appropriate for Spero-19 configuration)
        """
        if self.wavelength == [] or self.Pixels == [] :
            return None         # Nothing to work with
        if len(self.wavelength) < 3 :
            return None         # need at least [first, second, last]
        Step = self.wavelength[1] - self.wavelength[0]
        Last = self.wavelength[-1]
        wn = self.wavelength[0]
        missing = []            # tally up wave numbers that we'd expect but are not found
        LastBeforeGap = None    # for the moment handle only a single gap
        IndexOfLastBeforeGap = None
        i=0
        while wn <= Last :
            if not wn in self.wavelength :
                missing.append(wn)
                if None == LastBeforeGap :
                    LastBeforeGap = wn-Step
                    IndexOfLastBeforeGap = i-1
            wn += Step
            i  += 1

        print(LastBeforeGap, IndexOfLastBeforeGap)
        if None == LastBeforeGap :
            return []           # No gap, nothing more to do
        if OnlyFindGaps :
            return missing      # that's all we need
        NewWavelengthLength = (self.Pixels.shape[2]+len(missing))
        NewShape = (self.Pixels.shape[0], self.Pixels.shape[1], NewWavelengthLength)
        print(NewShape)
        NewPixels = np.zeros( NewShape, self.Pixels.dtype)
        NewWavenumbers = NewWavelengthLength*[0.0] # fill in numbers, replace them later
        # return missing          # that's all we need
        # Insert the 'before' and 'after' bits
        NewWavenumbers[0:(IndexOfLastBeforeGap+1)] = self.wavelength[0:(IndexOfLastBeforeGap+1)]
        NewWavenumbers[(IndexOfLastBeforeGap+1):(IndexOfLastBeforeGap+len(missing))] = missing[:]
        NewWavenumbers[(IndexOfLastBeforeGap+len(missing)+1):] = self.wavelength[IndexOfLastBeforeGap+1:]
        # return NewWavenumbers
        # Insert existing slices in new places
        print(NewPixels.shape)
        NewPixels[:,:,0:(IndexOfLastBeforeGap+1)] = self.Pixels[:,:,0:(IndexOfLastBeforeGap+1)]
        NewPixels[:,:,(IndexOfLastBeforeGap+len(missing)):] = self.Pixels[:,:,IndexOfLastBeforeGap:]
        Delta = float(len(missing))
        NextIndex = IndexOfLastBeforeGap+len(missing)+1 # in new ordering, follows IndexOfLastBeforeGap
        # Linearly interpolate missing image values
        for k in range(IndexOfLastBeforeGap+1, NextIndex) :
            alpha = (k-IndexOfLastBeforeGap)/Delta
            NewPixels[:,:,k] = (1.0 - alpha)*NewPixels[:,:,IndexOfLastBeforeGap] + alpha*NewPixels[:,:,NextIndex]
        # Add new values to this object
        self.Pixels = NewPixels
        self.wavelength = NewWavenumbers
        self.bands = NewWavelengthLength
        return missing      # Done! (Rest of information is included in this object)

def AlternateDiff( Pixels ) :
    """ Pixels: input hypercube.
    Replace the slice[i] with the difference (slice[i] - slice[i-1])
    slice[0] is the mean of the rest of them, so the size of the hypercube is unchanged
    Returns an image hypercube of the same size and type."""
    Output = np.copy(Pixels)    # same size and type
    N = Output.shape[2]         # number of image planes
    for k in range(1,N) :
        Output[:,:,k] = Output[:,:,k] - Pixels[:,:,k-1]
    # Now put the mean into slice 0
    Output[:,:,0] = np.mean(Output[:,:,1:],2)
    return Output

def MergeEnviImages(Img1, Img2) :
    """Combine Img1 and Img2 along the wavenumber direction to form a new image.
    The images must be the same type and size (lines and bands) and the wavenumber ranges
    must be disjoint.  So, [1000, 1004, 1008] is mergable with [1100, 1104, 1108] but not
    with [1005, 1007, 1009].
    """
    ImgRet = EnviImage()
    # Test for conformance
    if type(Img1) != type(ImgRet) or type(Img2) != type(ImgRet) :
        print("? Not Envi images")
        return None             # not an Envi image
    if Img1.lines != Img2.lines or Img1.samples != Img2.samples :
        print("? Not the same size")
        return None             # images are different sizes
    if Img1.StoredAsTransmittance != Img2.StoredAsTransmittance :
        print("? Transmission/Reflection mismatch")
        return None             # different kind of values stored in voxels
    # Look at wavenumbers. Figure out which one is lower (ImgA), which one is higher (ImgB)
    # or if they are interleaved (error)
    ImgA=Img1;	ImgB=Img2;      # thank goodness these are just references!
    # Swap looking at minima
    if min(ImgB.wavelength) < min(ImgA.wavelength) :
        ImgB=Img1;	ImgA=Img2; # swap so ImgA has lower wavenumber range
    if min(ImgB.wavelength) < max(ImgA.wavelength) :
        print("? Wrong wavvelentgh order")
        return None             # wavenumbers interleave
    # Copy rest of fields (assuming they're OK)
    ImgRet.description = ImgA.description
    ImgRet.samples =  ImgA.samples
    ImgRet.lines =  ImgA.lines
    ImgRet.header_offset = 0
    ImgRet.data_type =  ImgA.data_type
    ImgRet.interleave =  ImgA.interleave
    ImgRet.sensor_type =  ImgA.sensor_type
    ImgRet.byte_order =  ImgA.byte_order
    ImgRet.StoredAsTransmittance = Img1.StoredAsTransmittance
    
    ImgRet.wavelength = ImgA.wavelength + ImgB.wavelength
    ImgRet.bands =  ImgA.bands + ImgB.bands
    ImgRet.Pixels = np.zeros([ImgA.samples, ImgA.lines, ImgRet.bands], ImgA.Pixels.dtype)
    ImgRet.Pixels[:,:,0:ImgA.bands] = ImgA.Pixels
    ImgRet.Pixels[:,:,ImgA.bands:] = ImgB.Pixels

    return ImgRet
    
    
Example = """description = { ChemVision File, File Name [Sample_2017_04_27(15_13_46)], Created [Thu Apr 27 15:13:51 2017]}
samples = 472
lines = 472
bands = 397
header offset = 0
file type = Spero Chemvision
data type = 4
interleave = bsq
sensor type = Daylight Solutions Spero
byte order = 0
wavelength units = cm-1
wavelength = { 1000, 1002, 1004, 1006, 1008, 1010, 1012, 1014, 1016, 1018, 1020, 1022, 1024, 1026, 1028, 1030, 1032, 1034, 1036, 1038, 1040, 1042, 1044, 1046, 1048, 1050, 1052, 1054, 1056, 1058, 1060, 1062, 1064, 1066, 1068, 1070, 1072, 1074, 1076, 1078, 1080, 1082, 1084, 1086, 1088, 1090, 1092, 1094, 1096, 1098, 1100, 1102, 1104, 1106, 1108, 1110, 1112, 1114, 1116, 1118, 1120, 1122, 1124, 1126, 1128, 1130, 1132, 1134, 1136, 1138, 1140, 1142, 1144, 1146, 1148, 1150, 1152, 1154, 1156, 1158, 1160, 1162, 1164, 1166, 1168, 1170, 1172, 1174, 1176, 1178, 1180, 1182, 1184, 1186, 1188, 1190, 1192, 1194, 1196, 1198, 1200, 1202, 1204, 1206, 1208, 1210, 1212, 1214, 1216, 1218, 1220, 1222, 1224, 1226, 1228, 1230, 1232, 1234, 1236, 1238, 1240, 1242, 1244, 1246, 1248, 1250, 1252, 1254, 1256, 1258, 1260, 1262, 1264, 1266, 1268, 1270, 1272, 1274, 1276, 1278, 1280, 1282, 1284, 1286, 1288, 1290, 1292, 1294, 1296, 1298, 1300, 1302, 1304, 1306, 1308, 1310, 1312, 1314, 1316, 1318, 1320, 1322, 1324, 1326, 1328, 1330, 1332, 1334, 1336, 1338, 1340, 1342, 1344, 1346, 1348, 1350, 1352, 1354, 1356, 1358, 1360, 1362, 1364, 1366, 1368, 1370, 1372, 1374, 1376, 1378, 1380, 1382, 1384, 1386, 1388, 1390, 1392, 1394, 1396, 1398, 1400, 1402, 1404, 1406, 1408, 1410, 1412, 1414, 1416, 1418, 1420, 1422, 1424, 1426, 1428, 1430, 1432, 1434, 1436, 1438, 1440, 1442, 1444, 1446, 1448, 1450, 1452, 1454, 1456, 1458, 1460, 1462, 1464, 1466, 1468, 1470, 1472, 1474, 1476, 1478, 1480, 1482, 1484, 1486, 1488, 1490, 1492, 1494, 1496, 1498, 1500, 1502, 1504, 1506, 1508, 1510, 1512, 1514, 1516, 1518, 1520, 1522, 1524, 1526, 1528, 1530, 1532, 1534, 1536, 1538, 1540, 1542, 1544, 1546, 1548, 1550, 1552, 1554, 1556, 1558, 1560, 1562, 1564, 1566, 1568, 1570, 1572, 1574, 1576, 1578, 1580, 1582, 1584, 1586, 1588, 1590, 1592, 1594, 1596, 1598, 1600, 1602, 1604, 1606, 1608, 1610, 1612, 1614, 1616, 1618, 1620, 1622, 1624, 1626, 1628, 1630, 1632, 1634, 1636, 1638, 1640, 1642, 1644, 1646, 1648, 1650, 1652, 1654, 1656, 1658, 1660, 1662, 1664, 1666, 1668, 1670, 1672, 1674, 1676, 1678, 1680, 1682, 1684, 1686, 1688, 1690, 1692, 1694, 1696, 1698, 1700, 1702, 1704, 1706, 1708, 1710, 1712, 1714, 1716, 1718, 1720, 1722, 1724, 1726, 1728, 1730, 1732, 1734, 1736, 1738, 1740, 1742, 1744, 1746, 1748, 1750, 1752, 1754, 1756, 1758, 1760, 1762, 1764, 1766, 1768, 1770, 1772, 1774, 1776, 1778, 1780, 1782, 1784, 1786, 1788, 1790, 1792}"""
