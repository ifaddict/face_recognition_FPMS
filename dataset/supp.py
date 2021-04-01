import os
import glob

i = 0

for file in glob.glob(r"Pictures/*.jpg"):
    os.remove(file)
    i +=1
    if i == 1600:
        break