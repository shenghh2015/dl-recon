function working_gen_data_60D_noIC_Nov21()

addpath(genpath('./toolbox_image'))
output_dirname = 'dataset_60D_noIC/';
mkdir(output_dirname);
noise=0;
theta = (-30:29)*pi/180;
tv_param=0;
generate_v10_Analytical_Noise(theta,noise,output_dirname,tv_param);