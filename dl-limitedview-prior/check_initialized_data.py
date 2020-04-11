import matplotlib.pyplot as plt
import glob
import pickle
import numpy as np

## this script is used to visualize the training and testing samples

#folder='/home/shenghua/DL-recon/dl-limitedview-prior/datasets/v6_train/'
#folder='/home/shenghua/DL-recon/dl-limitedview-prior/datasets/v6_test/'
# folder='/home/shenghua/dl-recon-shh/dl-limitedview-prior/datasets/v6_testFinal/'
folder='/home/shenghua/dl-recon-shh/dl-limitedview-prior/datasets/vEXP3.11-1_train_AD/'
files=glob.glob(folder+'*.pkl')

imgs=[]

for i in range(len(files)):
	file=files[i]
# 	if i<10:
	with open(file,'rb') as f:
		data = pickle.load(f)
		imgs.append(data)
			
plt.ion()
fig=plt.figure()
for i in range(len(imgs)):
# 	i = 1
	rd = random.sample(range(len(imgs)),1)[0]
	plt.clf()
	image=imgs[rd]['X']
	shp=image.shape
	# image=image.reshape((shp[0],shp[1]))
	ax=fig.add_subplot(1,2,1)
	cax = ax.imshow(np.squeeze(imgs[rd]['X']))
	fig.colorbar(cax)
	ax.set_title('LS-image')
	ax=fig.add_subplot(1,2,2)
	cax = ax.imshow(np.squeeze(imgs[rd]['Y']))
	ax.set_title('Orginal image')
	fig.colorbar(cax)
	plt.pause(2)
