function working_generate_datasets_IC_Nov21()
% WORKING_GENERATE_DATASETS_NO_IC_NOISE_AUG9 ... 
%  
%  


% add the toolbox_image
addpath(genpath('./toolbox_image'))
%output_dirname = 'dataset_v5_50D_IC_Noise_TVC/';mkdir(output_dirname);
output_dirname = 'dataset_v5_100D_noIC_Noise/';mkdir(output_dirname);
noise=.02;
theta = (-50:49)*pi/180;
tv_param=0;
generate_v10_Analytical_Noise(theta,noise,output_dirname,tv_param);