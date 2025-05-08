import os, sys, time
import numpy as np
import h5py
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
sys.path.insert(1, './FourCastNet/') # insert code repo into path

"""
Run this script like:
python dist_inf_base.py --prediction_length --num_gpus

where prediction length is the number of autoregressive steps conducted by the distributed inference.

"""

# you may need to
# !pip install ruamel.yaml einops timm
# (or conda install)

from utils.YParams import YParams
from networks.afnonet import AFNONet

from constants import VARIABLES
from proj_utils import load_model, inference, lat, latitude_weighting_factor, weighted_rmse_channels
#from distributed_utils import inference_ensemble
from distributed_inference import inference_ensemble

PLOT_INPUTS = False # to get a sample plot
COMPILE = False # to use torch.compile()

# DO THIS WITHIN YOUR SCRATCH AND SET PATH
# wget https://portal.nersc.gov/project/m4134/ccai_demo.tar
# tar -xvf ccai_demo.tar
# rm ccai_demo.tar

base_path = "/pscratch/sd/a/ac5114/ML_class/" # update to yours

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
    project="weather-forecast-inference",entity="jhalpern-columbia-university",    # Specify your project
    config={                         # Track hyperparameters and metadata
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
prediction_length = 20 # number of steps (x 6 hours)

# which field to track for visualization
field = 'u10'
idx_vis = VARIABLES.index(field) # also prints out metrics for this field

# get prediction length slice from the data
print('Loading inference data')
print('Inference data from {}'.format(data_file))
data = h5py.File(data_file, 'r')['fields'][ic:(ic+prediction_length*dt):dt,in_channels,0:img_shape_x]
print(data.shape)
print("Shape of data = {}".format(data.shape))

data = (data - means)/stds # standardize the data
    
data = data[np.newaxis, :, :, :]

ensemble_size = sys.argv[-1]

ensemble_init = data.repeat(ensemble_size,axis=0)
ensemble_init = torch.tensor(ensemble_init, device=device, dtype=torch.float)

epsilon = 1e-3  # perturbation magnitude
ensemble_init += epsilon * torch.randn_like(ensemble_init.clone().detach())

# Set the prediction length (number of autoregressive steps)
prediction_length = sys.argv[1]

# Run the ensemble inference and measure the performance
inference_results = inference_ensemble(ensemble_init, model, prediction_length, idx_vis, params, device = device,img_shape_x = img_shape_x, img_shape_y = img_shape_y, std = std, m =m, field = field)
