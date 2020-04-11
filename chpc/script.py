import os

## dataset 1: dataset_60D_noIC

dataset = '../xct-parallelbeam-matlab/dataset_60D_noIC'
model_folder = '../dl-limitedview-prior/training_runs/EXP3.11-/stage9_model/'
save_dir = 'experiment_AD/'
tmp_save = 852
out_folder = 'out_folder'
num_add = 1
theta = 60
command = 'python3 generate_AD_dataset.py --dataset '+dataset+' --depth 10 --nb_filters 32 --model_folder '+model_folder+' --save_dir '+save_dir+' --num_add '+str(num_add)+' --theta '+str(theta)+' --residualNet True --tmp_save '+str(tmp_save)+' --TVConstraint 0'
print(command)
os.system(command)
