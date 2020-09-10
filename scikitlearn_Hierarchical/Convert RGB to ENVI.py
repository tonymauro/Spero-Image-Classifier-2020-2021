from ENVI_Files import Envi

print("Convert RGB to ENVI:")

# # Need to create the object before we can do anything with it
ei = Envi.EnviImage()

text = input("Enter Image Path (Start with * to enter custom directory): ")
if text.startswith("*"):
    image = text[1:]
else:
    image = "ENVI_Files\\Images\\" + text

# # ReadImage returns the values originally read (8 bit integers)
# # Assign to a variable to keep it from printing
im = ei.ReadImage(image)
name = input("Enter name to save as (No File Extension!): ")

# # Save this in ENVI format (note no extension)
ei.Write("ENVI_Files\\Images\\" + name)
