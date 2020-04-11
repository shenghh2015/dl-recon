# scp -r bmkelly@turing.seas.wustl.edu:/home/bmkelly/dl-limitedview-prior/generate_AD_dataset.py
from keras.callbacks import Callback
import helper_functions as hf
import CNN_generator as cg
import numpy as np
import argparse
import random
import os
import math


# python3 generate_AD_dataset.py --dataset 19 --save_dir v19_AD/

# old_env = os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_dim1", type=int, default=16)
    parser.add_argument("--f_dim2", type=int, default=1)
    parser.add_argument("--f_dim3", type=int, default=8)
    parser.add_argument("--nb_filters", type=int, default=325)
    parser.add_argument("--depth",type=int,default=3)
    #parser.add_argument("--dataset_folder",type=str,default='13')
    parser.add_argument("--dataset",type=str,default='13')
    parser.add_argument("--num_stacks",type=int,default=1)
    parser.add_argument("--tmp_save",type=int,default=0)
    parser.add_argument("--num_add",type=int,default=50)
    parser.add_argument("--model_folder",type=str,default='trained_v19')
    parser.add_argument("--save_dir",type=str,default='tmp/')
    parser.add_argument("--residualNet",type=bool,default=False)
    parser.add_argument("--dimension",type=int,default=256)
    parser.add_argument("--theta",type=int,default=50)
    parser.add_argument("--TVConstraint",type=int,default=0)
    args = parser.parse_args()
    return args

args = get_args()
print(args)

name = args.model_folder
max_index=9000
theta = args.theta
data_dirname = args.dataset
H_DIRNAME = '../xct-parallelbeam-matlab/system-matrix/'

# model_just_weights = hf.load_trained_CNN(load_weights=True,name=name+'/best_simple',folder='')
# model_larger = cg.residual_projectionNet2(depth=args.depth,input_shape=(args.dimension,args.dimension,1),k1=args.f_dim1,k2=args.f_dim2,k3=args.f_dim3,nb_filters=args.nb_filters)
model_larger = hf.load_trained_CNN(load_weights=True,name=name+'/best_simple',folder='')

print('Model loaded!')
# scp -r bmkelly@deeplearning.seas.wustl.edu:/home/bmkelly/xct-parallelbeam-matlab/system-matrix xct-parallelbeam-matlab
# scp -r bmkelly@deeplearning.seas.wustl.edu:/home/bmkelly/xct-parallelbeam-matlab/dataset_v19_100_noRI_scale_nonneg_noInvCrime_PoissonNoise  xct-parallelbeam-matlab


save_file = args.save_dir + str(args.tmp_save)+ '.pkl' 
# Generate datapoints, save to pkl file
def generate_AD(save_file = save_file, model_larger=model_larger,num_add=args.num_add,max_index=max_index,theta=theta,data_dirname=data_dirname,H_DIRNAME=H_DIRNAME,tmp_save=args.tmp_save):
    Y_train = []
    X_train = []
    print(save_file)
    for k in range(num_add):
        index = tmp_save*num_add + k
        print('sample '+str(index))
        skip=False
        try:
            H,g,f_true,f_recon = hf.load_H_g_target(version=2,index=index,dirname=data_dirname,H_DIRNAME=H_DIRNAME,theta=50)
        except Exception:
            print('couldnt find index: ' + str(index) + '.At location: ' + data_dirname + '/' + str(index) + ', H_dirname: ' + H_DIRNAME)
            skip=True
        if not skip:
            min_cutoff = math.log10(.001)
            max_cutoff = math.log10(.001)
            cutoff=math.pow(10,(min_cutoff + (max_cutoff-min_cutoff)*np.random.random())) #max cutoff is .05, min is .005, sampling in log space
            niters=100000 # max number of iterations --not really used, ignore
            njumps=16 # number of projections
            if args.TVConstraint == 0:
            	f_deep,psnr_save,projection_times_save,magnitudes_save,peak_saves,peak_saves_iterations,data_fidelity_loss = hf.Projected_PLS(model_larger,njumps=njumps,cutoff=cutoff,niters=niters,version=2,theta=theta,index=index,display=False,data_dirname=data_dirname,linked=not args.residualNet,H_DIRNAME=H_DIRNAME)
            else:
            	f_deep,psnr_save,projection_times_save,magnitudes_save,peak_saves,peak_saves_iterations,data_fidelity_loss = hf.Projected_PLS(model_larger,njumps=njumps,cutoff=cutoff,niters=niters,version=2,theta=theta,index=index,display=False,data_dirname=data_dirname,linked=not args.residualNet,H_DIRNAME=H_DIRNAME,TVConstraint=True)
            
            for j in range(len(peak_saves)):
                Y_inter = []
                X_inter = []
                X_inter.append(peak_saves[j].reshape(256,256,1))
                Y_inter.append(f_true.reshape(256,256,1))
                X_inter = np.asarray(X_inter)
                Y_inter = np.asarray(Y_inter)
                X_train.append(X_inter[0])
                Y_train.append(Y_inter[0]) # hopefully this is the same as Y_train[index]...


    # Save Data
    import pickle
    data = {}
    data['X_train'],data['y_train'] = X_train,Y_train
    with open(save_file, 'wb') as f:
        pickle.dump(data,f)

generate_AD()







# EoF
