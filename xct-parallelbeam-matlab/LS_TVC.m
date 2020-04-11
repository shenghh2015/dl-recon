function LS_TVC(theta,H_prefix_file_path,save_fname,initialization_fname)
% LS_TVC ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 19-Jun-2017 18:00:21 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : LS_TVC.m 

NX = 256;
STEP_SIZE=.5;
gamma = 600;
cost_cutoff = .001;
VERBOSE=0;
%% Load H
H= load_H_matrix(H_prefix_file_path,str2num(theta));

%% Load Recon data
% fid = fopen([mf 'measdata' num2str(index), '.dat'], 'rb');
% g = fread(fid,'float');
% fclose(fid);

load(initialization_fname);
input_data = double(input_data);
g = double(g);

data.g = g(:);
data.H = H;


recon = fistatv2d(@cost_func_xray_H, input_data, data, ...
            STEP_SIZE, 0, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'min_rel_cost_diff', cost_cutoff,'max_iter',1000,'proj_op','nonneg', ...
            'tv_projection_gamma',gamma);


save(save_fname,'recon');


 
% ===== EOF ====== [LS_TVC.m] ======  
