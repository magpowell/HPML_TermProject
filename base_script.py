import os, sys, time
import numpy as np
import h5py
import torch
import torchvision
import torch.nn as nn
import torch.quantization
import matplotlib.pyplot as plt
import wandb
sys.path.insert(1, './FourCastNet/') # insert code repo into path

"""
*******************************
Usage:
Run this script by passing arguments to the command line as follows:

python base_script.py --torch.compile --quantization --num-gpus --prediction-length --variable


--torch.compile: Boolean, choose if running in compile mode for speedup
--quantization: Boolean, choose if you'd like to run with quantized linear layer weights
--num-gpus: Int, number of GPUs to use for distributed inference
--prediction-length: Int, number of timesteps for autoregressive loop
--variable: String, variable name you'd like to calculate. Options are:
    variables = ['u10' (10 metre zonal wind speed m s-1),
             'v10' (10 metre meridional wind speed m s-1),
             't2m' (2 metre temperature K),
             'sp' (Surface pressure Pa),
             'msl' (Mean sea level pressure Pa),
             't850' (temperature at the 850 hPa pressure level K),
             'u1000' (zonal wind at 1000 mbar pressure surface m s-1),
             'v1000' (meridional wind at 1000 mbar pressure surface m s-1),
             'z1000' (vertical wind at 1000 mbar pressure surface m s-1),
             'u850' (zonal wind at 850 mbar pressure surface m s-1),
             'v850' (meridional wind at 850 mbar pressure surface m s-1),
             'z850' (vertical wind at 850 mbar pressure surface m s-1),
             'u500' (zonal wind at 500 mbar pressure surface m s-1),
             'v500' (meridional wind at 500 mbar pressure surface m s-1),
             'z500' (vertical wind at 500 mbar pressure surface m s-1),
             't500' (temperature wind at 500 mbar pressure surface K),
             'z50'  (geopotential height at 50 hPa),
             'r500' (relative humidity at 500 mbar pressure surface),
             'r850' (relative humidity at 850 mbar pressure surface),
             'tcwv' (total column water vapor kg m-2)]

*******************************
"""


# you may need to
# !pip install ruamel.yaml einops timm
# (or conda install)

from utils.YParams import YParams
from networks.afnonet import AFNONet

from constants import VARIABLES
from proj_utils import load_model, inference, lat, latitude_weighting_factor, weighted_rmse_channels
from quantize import replace_linear_with_target_and_quantize, W8A16LinearLayer, model_size

PLOT_INPUTS = False # to get a sample plot
COMPILE = bool(sys.argv[1]) # to use torch.compile()
QUANTIZE = bool(sys.argv[2]) # to use post-training quantization
QDTYPE = torch.int8

num_gpus = int(sys.argv[3])

prediction_length = int(sys.argv[4]) # number of steps (x 6 hours)

# which field to track for visualization
field = sys.argv[-1]

# DO THIS WITHIN YOUR SCRATCH AND SET PATH
# wget https://portal.nersc.gov/project/m4134/ccai_demo.tar
# tar -xvf ccai_demo.tar
# rm ccai_demo.tar

base_path = "/pscratch/sd/m/mpowell/hpml/" # update to yours

# data and model paths
data_path = f"{base_path}ccai_demo/data/FCN_ERA5_data_v0/out_of_sample"
data_file = os.path.join(data_path, "2018.h5")
model_path = f"{base_path}ccai_demo/model_weights/FCN_weights_v0/backbone.ckpt"
global_means_path = f"{base_path}ccai_demo/additional/stats_v0/global_means.npy"
global_stds_path = f"{base_path}ccai_demo/additional/stats_v0/global_stds.npy"
time_means_path = f"{base_path}ccai_demo/additional/stats_v0/time_means.npy"
land_sea_mask_path = f"{base_path}ccai_demo/additional/stats_v0/land_sea_mask.npy"

os.environ["WANDB_NOTEBOOK_NAME"] = './base_script.py' # this will be the name of the notebook in the wandb project database
wandb.login()
run = wandb.init(
    project="weather-forecast-inference",    # Specify your project
    config={                         # Track hyperparameters and metadata
            "quantize": QUANTIZE,
            "compile": COMPILE, 
            "qdtype": QDTYPE,
            "gpus": num_gpus,
            "variable": field
    },
)

# default
config_file = "./FourCastNet/config/AFNO.yaml"
config_name = "afno_backbone"
params = YParams(config_file, config_name)
print("Model architecture used = {}".format(params["nettype"]))

if PLOT_INPUTS:
    sample_data = h5py.File(data_file, 'r')['fields']
    print('Total data shape:', sample_data.shape)
    timestep_idx = 0
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    for i, varname in enumerate(['u10', 't2m', 'z500', 'tcwv']):
        cm = 'bwr' if varname == 'u10' or varname == 'z500' else 'viridis'
        varidx = VARIABLES.index(varname)
        ax[i//2][i%2].imshow(sample_data[timestep_idx, varidx], cmap=cm)
        ax[i//2][i%2].set_title(varname)
    fig.tight_layout()

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
    wandb.log({"model_size_reduction":final_size/init_size}) 

if COMPILE:
    model = torch.compile(model, backend = 'inductor')

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

idx_vis = VARIABLES.index(field) # also prints out metrics for this field

# get prediction length slice from the data
print('Loading inference data')
print('Inference data from {}'.format(data_file))
data = h5py.File(data_file, 'r')['fields'][ic:(ic+prediction_length*dt):dt,in_channels,0:img_shape_x]
print(data.shape)
print("Shape of data = {}".format(data.shape))


# Announce variable name:
print('Running inference for variable '.format(field))

# run inference
data = (data - means)/stds # standardize the data

if num_gpus == 1:
    data = torch.as_tensor(data).to(device, dtype=torch.float) # move to gpu for inference

    total_time, avg_time, acc_cpu, rmse_cpu, predictions_cpu, targets_cpu = inference(data, model, prediction_length, idx=idx_vis,
                                                                                  params = params, device = device, 
                                                                                  img_shape_x = img_shape_x, img_shape_y = img_shape_y, std = std, m =m, field = field)

if num_gpus > 1:
    print("Running with Num GPUs = {}".format(num_gpus))
    from distributed_inference import inference_ensemble

    data = data[np.newaxis, :, :, :]
    ensemble_init = np.tile(data, (num_gpus, 1, 1, 1, 1))
    epsilon = 1e-3 

    random_values = np.random.randn(num_gpus)
    random_values = random_values[:, np.newaxis, np.newaxis, np.newaxis]
    ensemble_init *= (random_values * epsilon)

    # Run the ensemble inference and measure the performance
    inference_results = inference_ensemble(ensemble_init, model, prediction_length, idx_vis, params, device = device,img_shape_x = img_shape_x, img_shape_y = img_shape_y, std = std, m =m, field = field, num_gpus = num_gpus)
