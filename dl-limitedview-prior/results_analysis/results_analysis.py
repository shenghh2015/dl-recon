import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pickle
from skimage.measure import compare_ssim as ssim

root_folder = '/home/shenghua/dl-recon-shh/dl-limitedview-prior/datasets/'
# ls_folder = root_folder + 'v6_testFinal/'
ls_folder = '/home/shenghua/dl-recon-shh/xct-parallelbeam-matlab/dataset_60D_noIC/'
pls_tv_folder = '/home/shenghua/dl-recon-shh/xct-parallelbeam-matlab/dataset_60D_noIC_TV/'
sp_folder = root_folder + 'vexperiment-12.13-0_AD'
dl_folder = root_folder + 'vexperiment-12.13-4_AD'

def read_data(dirname, fn_prefix, ind, version=0, AD = False, dl = False, img_size= 256):
	"""Reads binary single-precision floating-point data from disk

	Arguments:
	  dirname - directory name where the data is located
	  fn_prefix - filename prefix, assumes the filename is of the
		  form: <FN_PREFIX><IND>.dat
	  ind - the index of the sample

	Returns:
	  A 1D float32 numpy array containing the data read from disk
	"""
	filename = os.path.join(dirname, '%s%d.pkl' % (fn_prefix, ind))
# 	print(filename)
	f= open(filename, 'rb')
	data = pickle.load(f)
	recon = None
	true = None
	if AD == True:
		recon_list = data['X_train']
		true_list = data['y_train']
		if dl == True:
			true = true_list[-1]
			recon = recon_list[-1]
		else:
			true = true_list[1]
			recon = recon_list[1]
	else:
		recon = data['X']
		true = data['Y']

	return recon, true

def read_recon_list(dirname, fn_prefix, ind, version=0, AD = False, dl = False, img_size= 256, nb_proj = 6):
	"""Reads binary single-precision floating-point data from disk

	Arguments:
	  dirname - directory name where the data is located
	  fn_prefix - filename prefix, assumes the filename is of the
		  form: <FN_PREFIX><IND>.dat
	  ind - the index of the sample

	Returns:
	  A 1D float32 numpy array containing the data read from disk
	"""
	filename = os.path.join(dirname, '%s%d.pkl' % (fn_prefix, ind))
# 	print(filename)
	f= open(filename, 'rb')
	data = pickle.load(f)
	recon_ls = []
	true_ls = []
	if AD == True:
		recon_list = data['X_train']
		true_list = data['y_train']
# 		print(len(recon_list))
		if dl == True:
# 			for k in range(10):
# 				idx = k*nb_proj*2+11
# 				true = true_list[idx]
# 				recon = recon_list[idx]
# 				recon_ls.append(recon)
# 				true_ls.append(true)
			deep_idx_list = select_deep_idx(recon_list, true_list, img_size = img_size, nb_proj = nb_proj)
			for idx in deep_idx_list:
				recon_ls.append(recon_list[idx])
				true_ls.append(true_list[idx])
		else:
			for k in range(10):
				idx = k*nb_proj*2+1
				true = true_list[idx]
				recon = recon_list[idx]
				recon_ls.append(recon)
				true_ls.append(true)

	return recon_ls, true_ls

def read_float_data(dirname, fn_prefix, ind):
	"""Reads binary single-precision floating-point data from disk

	Arguments:
	  dirname - directory name where the data is located
	  fn_prefix - filename prefix, assumes the filename is of the
		  form: <FN_PREFIX><IND>.dat
	  ind - the index of the sample

	Returns:
	  A 1D float32 numpy array containing the data read from disk
	"""
	filename = os.path.join(dirname, '%s%d.dat' % (fn_prefix, ind))
	return np.fromfile(filename, dtype=np.float32)
	
def read_recon_true(dirname, ind, img_size = 256):
	true = read_float_data(dirname, 'img', ind).reshape(256,256)
	recon = read_float_data(dirname, 'recon', ind).reshape(256,256)
	return recon, true

def select_deep_idx(recon_list, true_list, img_size = 256, nb_proj = 6):
	flat_recon_arr = np.array(recon_list).reshape(len(recon_list),-1)
	flat_true_arr = np.array(true_list).reshape(len(true_list),-1)
	mse_arr = np.mean((flat_recon_arr- flat_true_arr)**2, axis =1)
	deep_list = []
	for i in range(len(recon_list)):
		if i%(2*nb_proj) == 0:
			select_idx = np.argmin(mse_arr[i:i+2*nb_proj])+i
			deep_list.append(select_idx)
	return deep_list

# nb_final_test = 500
# nb_test = 1000
# nb_train = 6400
nb_final_test = 100
nb_test = 1000
nb_train = 6400
## load the LS recon data
ls_recon_list = []
ls_true_list = []
for i in range(nb_final_test):
	idx = nb_train + nb_test + i
# 	recon, true = read_data(ls_folder, '', ind = idx)
	recon, true = read_recon_true(ls_folder, ind = idx)
	ls_recon_list.append(recon)
	ls_true_list.append(true)
ls_recon_arr = np.array(ls_recon_list)
ls_true_arr = np.array(ls_true_list)
ls_mse = np.mean((ls_recon_arr- ls_true_arr)**2)
ls_rmse = np.sqrt(ls_mse)
print(ls_recon_arr.shape)
print('RMSE->LS:'+str(ls_rmse))
# ave_mse = np.mean(np.apply_over_axes(np.sum, (ls_recon_arr- ls_true_arr)**2, [1,2]))

## load the PLS-TV recon data
tv_recon_list = []
tv_true_list = []
for i in range(nb_final_test):
	idx = nb_train + nb_test + i
# 	recon, true = read_data(ls_folder, '', ind = idx)
	recon, true = read_recon_true(pls_tv_folder, ind = idx)
	tv_recon_list.append(recon)
	tv_true_list.append(true)
tv_recon_arr = np.array(tv_recon_list)
tv_true_arr = np.array(tv_true_list)
tv_mse = np.mean((tv_recon_arr- tv_true_arr)**2)
tv_rmse = np.sqrt(tv_mse)
print(tv_recon_arr.shape)
print('RMSE->PLS-TV:'+str(tv_rmse))


## load the SP recon data
sp_recon_list = []
sp_true_list = []
recon_ls = []
true_ls = []
offset = int((nb_train+nb_test)/10)
nb_final_test = int(nb_final_test/10)
for i in range(nb_final_test):
	idx = i + offset
# 	print(idx)
	recon_ls, true_ls = read_recon_list(sp_folder, '', ind = idx, AD = True, dl = False, nb_proj = 17)
# 	print('recon list length:'+str(len(recon_ls)))
	sp_recon_list += recon_ls
	sp_true_list += true_ls
sp_recon_arr = np.array(sp_recon_list)
sp_true_arr = np.array(sp_true_list)
sp_mse = np.mean((sp_recon_arr- sp_true_arr)**2)
sp_rmse = np.sqrt(sp_mse)
print(sp_recon_arr.shape)
print('RMSE->SP:'+str(sp_rmse))

## load the DL recon data
dl_recon_list = []
dl_true_list = []
recon_ls = []
true_ls = []
for i in range(nb_final_test):
	idx = i + offset
	recon_ls, true_ls = read_recon_list(dl_folder, '', ind = idx, AD = True, dl = True, nb_proj = 17)
	dl_recon_list += recon_ls
	dl_true_list += true_ls
dl_recon_arr = np.array(dl_recon_list)
dl_true_arr = np.array(dl_true_list)
dl_mse = np.mean((dl_recon_arr- dl_true_arr)**2)
dl_rmse = np.sqrt(dl_mse)
print(dl_recon_arr.shape)
print('RMSE->DL:'+str(dl_rmse))

result_str = 'LS-NN:{0:.4f}\nPLS-TV:{1:.4f}\nSingle-pass:{2:.4f}\nDeep-guided:{3:.4f}'.format(ls_rmse, tv_rmse, sp_rmse, dl_rmse)
print(result_str)

## calculate average SSIM
ls_ssim_ls = []
tv_ssim_ls = []
sp_ssim_ls = []
dl_ssim_ls = []
nb_final_test = 100
for i in range(nb_final_test):
	ssim_ls = ssim(ls_true_arr[i,:,:], ls_recon_arr[i,:,:], data_range = ls_recon_arr[i,:,:].max() - ls_recon_arr[i,:,:].min())
	ssim_tv = ssim(ls_true_arr[i,:,:], tv_recon_arr[i,:,:], data_range = tv_recon_arr[i,:,:].max() - tv_recon_arr[i,:,:].min())
	ssim_sp = ssim(ls_true_arr[i,:,:], np.squeeze(sp_recon_arr[i,:,:]), data_range = sp_recon_arr[i,:,:].max() - sp_recon_arr[i,:,:].min())
	ssim_dl = ssim(ls_true_arr[i,:,:], np.squeeze(dl_recon_arr[i,:,:]), data_range = dl_recon_arr[i,:,:].max() - dl_recon_arr[i,:,:].min())
	ls_ssim_ls.append(ssim_ls)
	tv_ssim_ls.append(ssim_tv)
	sp_ssim_ls.append(ssim_sp)
	dl_ssim_ls.append(ssim_dl)
result_str = 'SSIM=>\nLS-NN:{0:.4f}\nPLS-TV:{1:.4f}\nSingle-pass:{2:.4f}\nDeep-guided:{3:.4f}'.format(np.mean(ls_ssim_ls), np.mean(tv_ssim_ls), np.mean(sp_ssim_ls), np.mean(dl_ssim_ls))
print(result_str)

plt.ion()
fig = plt.figure()
nb_final_test = 500
for i in range(nb_final_test):
	plt.clf()
	ax = fig.add_subplot(4,2,1)
	cax = ax.imshow(ls_recon_arr[i,:,:])
	ax.set_ylabel('LS-NN')
	ax.set_title('Recon image')
	fig.colorbar(cax)
	ax = fig.add_subplot(4,2,2)
	cax = ax.imshow(ls_true_arr[i,:,:])
	ax.set_title('True image')
	fig.colorbar(cax)
	ax = fig.add_subplot(4,2,3)
	cax = ax.imshow(tv_recon_arr[i,:,:])
	ax.set_ylabel('PLS-TV')
	fig.colorbar(cax)
	ax = fig.add_subplot(4,2,4)
	cax = ax.imshow(tv_true_arr[i,:,:])
	fig.colorbar(cax)
	ax = fig.add_subplot(4,2,5)
	cax = ax.imshow(sp_recon_arr[i,:,:,0])
	ax.set_ylabel('Single-pass')
	fig.colorbar(cax)
	ax = fig.add_subplot(4,2,6)
	cax = ax.imshow(sp_true_arr[i,:,:,0])
	fig.colorbar(cax)
	ax = fig.add_subplot(4,2,7)
	cax = ax.imshow(dl_recon_arr[i,:,:,0])
	ax.set_ylabel('Deep-guided')
	fig.colorbar(cax)
	ax = fig.add_subplot(4,2,8)
	cax = ax.imshow(dl_true_arr[i,:,:,0])
	fig.colorbar(cax)
	plt.pause(1)

plt.ion()
fig = plt.figure()
nb_final_test = 100
for i in range(nb_final_test):
# 	i = 10
	plt.clf()
	ax = fig.add_subplot(1,5,1)
	cax = ax.imshow(ls_true_arr[i,:,:])
	ax.set_title('True image')
	fig.colorbar(cax)
	ax = fig.add_subplot(1,5,2)
	cax = ax.imshow(ls_recon_arr[i,:,:])
	ax.set_title('LS-NN')
	ssim_ = ssim(ls_true_arr[i,:,:], ls_recon_arr[i,:,:], data_range = ls_recon_arr[i,:,:].max() - ls_recon_arr[i,:,:].min())
	ax.set_xlabel('RMSE:{0:0.4f}\nSSIM:{1:0.4f}'.format(np.mean((ls_true_arr[i,:,:]-ls_recon_arr[i,:,:])**2),ssim_))
	fig.colorbar(cax)
	ax = fig.add_subplot(1,5,3)
	cax = ax.imshow(tv_recon_arr[i,:,:])
	ax.set_title('PLS-TV')
	ssim_ = ssim(ls_true_arr[i,:,:], tv_recon_arr[i,:,:], data_range = tv_recon_arr[i,:,:].max() - tv_recon_arr[i,:,:].min())
	ax.set_xlabel('RMSE:{0:0.4f}\nSSIM:{1:0.4f}'.format(np.mean((ls_true_arr[i,:,:]-tv_recon_arr[i,:,:])**2),ssim_))
	fig.colorbar(cax)
	ax = fig.add_subplot(1,5,4)
	cax = ax.imshow(sp_recon_arr[i,:,:,0])
	ax.set_title('Single-pass')
	ssim_ = ssim(ls_true_arr[i,:,:], np.squeeze(sp_recon_arr[i,:,:]), data_range = sp_recon_arr[i,:,:].max() - sp_recon_arr[i,:,:].min())
	ax.set_xlabel('RMSE:{0:0.4f}\nSSIM:{1:0.4f}'.format(np.mean((ls_true_arr[i,:,:]-np.squeeze(sp_recon_arr[i,:,:]))**2),ssim_))
	fig.colorbar(cax)
	ax = fig.add_subplot(1,5,5)
	cax = ax.imshow(dl_recon_arr[i,:,:,0])
	ax.set_title('Deep-guided')
	ssim_ = ssim(ls_true_arr[i,:,:], np.squeeze(dl_recon_arr[i,:,:]), data_range = dl_recon_arr[i,:,:].max() - dl_recon_arr[i,:,:].min())
	ax.set_xlabel('RMSE:{0:0.4f}\nSSIM:{1:0.4f}'.format(np.mean((ls_true_arr[i,:,:]-np.squeeze(dl_recon_arr[i,:,:]))**2),ssim_))
	fig.colorbar(cax)
	plt.pause(1)
