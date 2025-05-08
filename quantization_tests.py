import os, sys, time
import numpy as np
import pandas as pd
import h5py
import torch
import torchvision
import torch.nn as nn
import torch.quantization
import matplotlib.pyplot as plt
import wandb
sys.path.insert(1, './FourCastNet/') # insert code repo into path

from utils.YParams import YParams
from networks.afnonet import AFNONet

from constants import VARIABLES
from proj_utils import load_model, inference, lat, latitude_weighting_factor, weighted_rmse_channels
from distributed_utils import inference_ensemble
from quantize import replace_linear_with_target_and_quantize, W8A16LinearLayer, model_size

PLOT_INPUTS = False # to get a sample plot
COMPILE = False # to use torch.compile()
QUANTIZE = True # to use post-training quantization
QDTYPE = torch.int8

base_path = "/pscratch/sd/j/jhalpern/hpml/" # update to yours

# data and model paths
data_path = f"{base_path}ccai_demo/data/FCN_ERA5_data_v0/out_of_sample"
data_file = os.path.join(data_path, "2018.h5")
model_path = f"{base_path}ccai_demo/model_weights/FCN_weights_v0/backbone.ckpt"
global_means_path = f"{base_path}ccai_demo/additional/stats_v0/global_means.npy"
global_stds_path = f"{base_path}ccai_demo/additional/stats_v0/global_stds.npy"
time_means_path = f"{base_path}ccai_demo/additional/stats_v0/time_means.npy"
land_sea_mask_path = f"{base_path}ccai_demo/additional/stats_v0/land_sea_mask.npy"

os.environ["WANDB_NOTEBOOK_NAME"] = './quantization_tests.py' # this will be the name of the notebook in the wandb project database
wandb.login()
run = wandb.init(
    project="weather-forecast-inference",    # Specify your project
    config={                         # Track hyperparameters and metadata
            "quantize": QUANTIZE,
            "compile": COMPILE, 
            "qdtype": QDTYPE
    },
)

# default
config_file = "./FourCastNet/config/AFNO.yaml"
config_name = "afno_backbone"
params = YParams(config_file, config_name)
print("Model architecture used = {}".format(params["nettype"]))

# import model
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# in and out channels: FourCastNet uses 20 input channels corresponding to 20 prognostic variables
in_channels = np.array(params.in_channels)
out_channels = np.array(params.out_channels)
params['N_in_channels'] = len(in_channels)
params['N_out_channels'] = len(out_channels)
params.means = np.load(global_means_path)[0, out_channels] # for normalizing data with precomputed train stats
params.stds = np.load(global_stds_path)[0, out_channels]
params.time_means = np.load(time_means_path)[0, out_channels]

# load the model
if params.nettype == 'afno':
    model = AFNONet(params).to(device)  # AFNO model
else:
    raise Exception("not implemented")
# load saved model weights
model = load_model(model, params, model_path)
model = model.to(device)

if QUANTIZE:
    param_size, buffer_size = model_size(model)
    init_size = param_size + buffer_size
    print(f"Initial model size: {(init_size) / (1024 ** 2):.2f} MB, {param_size / (1024 ** 2):.2f} MB (parameters), {buffer_size /(1024 ** 2):.2f} MB (buffers)")
    print(QDTYPE)
    replace_linear_with_target_and_quantize(model, W8A16LinearLayer, QDTYPE)
    param_size, buffer_size = model_size(model)
    final_size = param_size + buffer_size
    print(f"Final model size: {(final_size) / (1024 ** 2):.2f} MB, {param_size / (1024 ** 2):.2f} MB (parameters), {buffer_size /(1024 ** 2):.2f} MB (buffers)")
    model_size_reduction = final_size / init_size
    wandb.log({"model_size_reduction":model_size_reduction}) 
else:
    model_size_reduction = 1

# move normalization tensors to gpu
# load time means: represents climatology
img_shape_x = 720
img_shape_y = 1440

# means and stds over training data
means = params.means
stds = params.stds

# load climatological means
time_means = params.time_means # temporal mean (for every pixel)
m = torch.as_tensor((time_means - means)/stds)[:, 0:img_shape_x]
m = torch.unsqueeze(m, 0)
# these are needed to compute ACC and RMSE metrics
m = m.to(device, dtype=torch.float)
std = torch.as_tensor(stds[:,0,0]).to(device, dtype=torch.float)

print("Shape of time means = {}".format(m.shape))
print("Shape of std = {}".format(std.shape))

# setup data for inference
dt = 1 # time step (x 6 hours)
ic = 0 # start the inference from here
prediction_length = 40 # number of steps (x 6 hours)

# which field to track for visualization
field = 'tcwv'
idx_vis = VARIABLES.index(field) # also prints out metrics for this field

# get prediction length slice from the data
print('Loading inference data')
print('Inference data from {}'.format(data_file))
data = h5py.File(data_file, 'r')['fields'][ic:(ic+prediction_length*dt):dt,in_channels,0:img_shape_x]
print(data.shape)
print("Shape of data = {}".format(data.shape))

# run inference
data = (data - means)/stds # standardize the data
data = torch.as_tensor(data).to(device, dtype=torch.float) # move to gpu for inference

total_time, avg_time, acc, rmse, predictions_cpu, targets_cpu = inference(data, model, prediction_length, idx=idx_vis,
                                                                                  params = params, device = device, 
                                                                                  img_shape_x = img_shape_x, img_shape_y = img_shape_y, std = std, m =m, field = field)
file_path = "./quantization_tests.csv"
if file_path:
    df = pd.DataFrame(
        {
            "tot_time": [total_time],
            "avg_time": [avg_time],
            "avg_accuracy": [np.mean(acc)],
            "avg_rmse": [np.mean(rmse)],
            "inference_data": field,
            "quantized_model_size": model_size_reduction,
            "qdtype": [QDTYPE],
            "use_quantization": [QUANTIZE],
        }
    )

    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)
