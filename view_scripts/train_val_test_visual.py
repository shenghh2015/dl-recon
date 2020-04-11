import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np

# data folder
# data_folder = '/home/shenghua/dl-recon-shh/dl-limitedview-prior/datasets/v6_train'
# data_folder = '/home/shenghua/dl-recon-shh/dl-limitedview-prior/datasets/v6_test'
data_folder = '/home/shenghua/dl-recon-shh/dl-limitedview-prior/datasets/v6_testFinal'
data_class1 = data_folder.split('_')[-1]
data_class2 = data_folder.split('/')[-1].split('_')[-2]

files = glob.glob(data_folder+'/*.pkl')
print(len(files))

plt.ion()
fig = plt.figure()
nb_sample = 10

for i in range(nb_sample):
	plt.clf()
	f = open(files[i], 'rb')
	data = pickle.load(f)
	x_img = None
	y_img = None
	if data_class1 == 'train' or data_class2 == 'train':
		x_img = data['X']
		y_img = data['Y']
	elif data_class1 == 'test' or data_class2 == 'test':
		x_img = data['X']
		y_img = data['Y']
	elif data_class1 == 'testFinal' or data_class2 == 'testFinal':
		x_img = data['X']
		y_img = data['Y']
	ax = fig.add_subplot(1,2,1)
	cax = ax.imshow(x_img)
	ax.set_title('image: '+str(i))
	ax.set_xlabel('X image')
	fig.colorbar(cax)
	ax = fig.add_subplot(1,2,2)
	cax = ax.imshow(y_img)
	fig.colorbar(cax)
	ax.set_title('image: '+str(i))
	ax.set_xlabel('Y image')
	plt.pause(1)

data_folder = '/home/shenghua/dl-recon-shh/dl-limitedview-prior/datasets/vexperiment-11.231_AD'
data_folder = '/home/shenghua/dl-recon-shh/chpc/experiment_AD'
data_folder = '/home/shenghua/dl-recon-shh/chpc/v1'
files = glob.glob(data_folder+'/*.pkl')
print(len(files))
plt.ion()
fig = plt.figure()
nb_sample = 120

i = 2
f = open(files[i], 'rb')
data = pickle.load(f)
X_train = data['X_train']
Y_train = data['y_train']
for k in range(len(X_train)):
	if k%2 ==1:
		continue
	j = k +1
	plt.clf()
	x_img = X_train[k][:,:,0]
	y_img = Y_train[k][:,:,0]
	xq_img = X_train[j][:,:,0]
	yq_img = Y_train[j][:,:,0]
	dif_img = np.abs(x_img-xq_img)
	ax = fig.add_subplot(3,2,1)
	cax = ax.imshow(x_img)
	ax.set_title('X image')
	ax.set_ylabel('image:'+str(k))
	fig.colorbar(cax)
	ax = fig.add_subplot(3,2,2)
	cax = ax.imshow(y_img)
	fig.colorbar(cax)
	ax.set_title('Y image')
	ax = fig.add_subplot(3,2,3)
	cax = ax.imshow(xq_img)
	ax.set_ylabel('image:'+str(j))
	fig.colorbar(cax)
	ax = fig.add_subplot(3,2,4)
	cax = ax.imshow(yq_img)
	fig.colorbar(cax)
	ax = fig.add_subplot(3,2,5)
	cax = ax.imshow(dif_img)
	fig.colorbar(cax)
	plt.pause(2)

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
root_folder = os.path.expanduser('~/dl-recon-shh/xct-parallelbeam-matlab')
img_pattern = os.path.join(root_folder, 'dataset_60D_noIC', 'img*.dat')
meas_pattern = os.path.join(root_folder, 'dataset_60D_noIC', 'meas*.dat')
recon_pattern = os.path.join(root_folder, 'dataset_60D_noIC', 'recon*.dat')
img_files = glob.glob(img_pattern)
meas_files = glob.glob(meas_pattern)
recon_files = glob.glob(recon_pattern)

plt.ion()
fig = plt.figure()
i = 0
plt.clf()
image = np.fromfile(img_files[i],dtype=np.float32)
image = image.reshape(256,256)
ax = fig.add_subplot(1,3,1)
ax.set_title('Phantom')
cax = ax.imshow(image)
fig.colorbar(cax)
ax = fig.add_subplot(1,3,2)
meas = np.fromfile(meas_files[i],dtype=np.float32)
meas = meas.reshape(60,256)
ax.set_title('Measurement data')
cax = ax.imshow(meas)
fig.colorbar(cax)
ax = fig.add_subplot(1,3,3)
recon = np.fromfile(recon_files[i],dtype=np.float32)
ax.set_title('LS-NN recon image')
recon = recon.reshape(256,256)
cax = ax.imshow(recon)
fig.colorbar(cax)

fig = plt.figure()
i = 0
plt.clf()
image = np.fromfile(img_files[i],dtype=np.float32)
image = image.reshape(256,256)
ax = fig.add_subplot(1,2,2)
ax.set_title('True image')
cax = ax.imshow(image)
fig.colorbar(cax)
# ax = fig.add_subplot(1,2,2)
# meas = np.fromfile(meas_files[i],dtype=np.float32)
# meas = meas.reshape(60,256)
# ax.set_title('Measurement data')
# cax = ax.imshow(meas)
# fig.colorbar(cax)
ax = fig.add_subplot(1,2,1)
recon = np.fromfile(recon_files[i],dtype=np.float32)
ax.set_title('Artifact image')
recon = recon.reshape(256,256)
cax = ax.imshow(recon)
fig.colorbar(cax)





