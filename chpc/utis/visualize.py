import numpy as np
import os

def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

# plot and save the file
def plot_save(result_folder,file_name,rmse_list,fidelity_list):
    generate_folder(result_folder)
    f_out=os.path.join(result_folder,file_name)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10,4))
    ax = fig.add_subplot(1,2,1)
    rmse_arr = np.array(rmse_list)
    rmse_arr = np.sqrt(rmse_arr)
    rmse_list = rmse_arr.tolist()
    ax.plot(rmse_list,'b-',linewidth=1.3)
    ax.set_title('MRSE over iterations')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iterations')
    ax.legend(['RMSE'], loc='upper right')
    ax = fig.add_subplot(1,2,2)
    ax.plot(fidelity_list,'r-',linewidth=1.3)
    ax.set_title('Cost over iterations')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iterations')
    ax.legend(['Cost'], loc='upper right')    
#     ax.plot(acc,'b-', linewidth=1.3)
#     ax.plot(val_acc,'r-',linewidth=1.3)
#     ax.set_title('Model Acc')
#     ax.set_ylabel('Accuracy')
#     ax.set_xlabel('epoches')
#     ax.legend(['train acc', 'test acc'], loc='upper left')   
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(f_out, dpi=80)