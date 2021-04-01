import PIL
from PIL import Image
import glob

for filename in glob.iglob(r'*.jpg', recursive=True):
	im = Image.open(filename)
	out = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
	out.save('flip_lr_' + filename)
	out = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
	out.save('flip_ud_' + filename)
	out = im.transpose(PIL.Image.FLIP_LEFT_RIGHT).transpose(PIL.Image.FLIP_TOP_BOTTOM)
	out.save('flip_udlr_' + filename)