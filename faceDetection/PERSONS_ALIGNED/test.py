import numpy as np
from PIL import Image, ImageOps

image_size = 182

#la bonne image
im1 = Image.open("2.png")
im1 = ImageOps.pad(im1, (image_size,)*2)
img1 = np.array(im1)
print("la bonne image : ", img1)


im2 = Image.open("4.png")
im2 = ImageOps.pad(im2, (image_size,)*2)
img2 = np.array(im2)
print("la mauvaise image : ", img2)