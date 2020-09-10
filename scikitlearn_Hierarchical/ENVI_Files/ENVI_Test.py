import numpy as np
import matplotlib.pyplot as plt
from ENVI_Files import Envi

# # Need to create the object before we can do anything with it
# ei = Envi.EnviImage()
# # ReadImage returns the values originally read (8 bit integers)
# # Assign to a variable to keep it from printing
# im = ei.ReadImage('Images\\miss')
# # Display one of the planes of the image
# plt.imshow(ei.Pixels[:,:,0])
# plt.show()
# # Save this in ENVI format (note no extension)
# ei.Write('beach')
# # Now, re-read it to verify we got it right
# # First create another object
# ei2 = Envi.EnviImage()
# # Read the saved date.  It is important that the arguments beyond the file
# # name appear as they do, lest the absorbtion transformation be mis-applied
# ei2.Read('beach', True, False, False)
# # Check that we completed the round-trip correctly
# plt.imshow(ei2.Pixels[:,:,0])
# plt.show()

# To read the data Spero produces (Held in TestFile)
ei3 = Envi.EnviImage()
# Defaults are reasonable for images Spero produces
ei3.Read('Images\\miss')
# Check out the size of the image hypercube
Out = ei3.Pixels.shape
Out[20]: (480, 480, 1081)
# Look at one of the images
plt.imshow(ei3.Pixels[:,:,0])
plt.show()
# Look at the spectrum at a single pixel
plt.plot(ei3.wavelength, ei3.Pixels[200,200,:])
plt.show()