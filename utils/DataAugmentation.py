import torchvision.transforms as transforms
import os
import PIL.Image as Image
import matplotlib.pyplot as plt

t1 = transforms.ColorJitter(brightness=(1,1.5), contrast=(1,1.5), saturation=(1,1.5))
#t2 = transforms.RandomAffine(0, translate=(0.05,0.05))
t = transforms.Compose([t1])


path = os.getcwd()
patterns_path = path+'/zoomed_out'
patterns = os.listdir(patterns_path)
for pattern in patterns:
	img_path = patterns_path+'/'+pattern
	img = Image.open(img_path)
	
	for i in range(1,3):
		img_ = t(img)
		print(patterns_path+'/'+pattern[:-4]+'_{}'.format(i)+pattern[-4:])
		img_.save(patterns_path+'/'+pattern[:-4]+'_{}'.format(i)+pattern[-4:])
