import os, sys, time
import numpy as np
import h5py
import torch
import torch.nn as nn
import click
import wandb

sys.path.insert(1, './FourCastNet/') # insert code repo into path
os.environ["ACCELERATE_USE_TP"] = "false"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
config_file = "./FourCastNet/config/AFNO.yaml"
config_name = "afno_backbone"

import torch.distributed as dist
from utils.YParams import YParams

from constants import VARIABLES
from proj_utils import make_ensemble_data, inference, load_data_and_model
from quantize import replace_linear_with_target_and_quantize, W8A16LinearLayer, model_size

QDTYPE = torch.int8

@click.command()
@click.option("--compile", is_flag=True, help="torch.compile()")
@click.option(
    "--quantize", is_flag=True, help="quantize)"
)
@click.option(
    "--ensemble_size",
    type=int,
    default = 1,
    help="file path for csv with results",
)
@click.option(
    "--field",
    default="u10",
    type=str,
    help="field of interest",
)
@click.option(
    "--base_path",
    default="/pscratch/sd/m/mpowell/hpml/",
    type=str,
    help="where is fourcast net data",
)
@click.option(
    "--file_path",
    type=str,
    help="file path for csv with results",
)
def main(compile, quantize, ensemble_size, field, base_path, file_path):

    params = YParams(config_file, config_name)
    print("Model architecture used = {}".format(params["nettype"]))

    model, data_file, in_channels, device, params = load_data_and_model(
        base_path, params
    )

    if compile:
        model = torch.compile(model, backend="inductor")

    # move normalization tensors to gpu
    # load time means: represents climatology
    img_shape_x = 720
    img_shape_y = 1440

    # means and stds over training data
    means = params.means
    stds = params.stds

    # load climatological means
    time_means = params.time_means  # temporal mean (for every pixel)
    m = torch.as_tensor((time_means - means) / stds)[:, 0:img_shape_x]
    m = torch.unsqueeze(m, 0)
    # these are needed to compute ACC and RMSE metrics
    m = m.to(device, dtype=torch.float)
    std = torch.as_tensor(stds[:, 0, 0]).to(device, dtype=torch.float)

    # setup data for inference
    dt = 1  # time step (x 6 hours)
    ic = 0  # start the inference from here
    prediction_length = 20  # number of steps (x 6 hours

    if quantize:
        param_size, buffer_size = model_size(model)
        init_size = param_size + buffer_size

        print(f"Initial model size: {(init_size) / (1024 ** 2):.2f} MB, {param_size / (1024 ** 2):.2f} MB (parameters), {buffer_size /(1024 ** 2):.2f} MB (buffers)")
        print(QDTYPE)

        replace_linear_with_target_and_quantize(model, W8A16LinearLayer, QDTYPE)
        param_size, buffer_size = model_size(model)
        final_size = param_size + buffer_size

        print(f"Final model size: {(final_size) / (1024 ** 2):.2f} MB, {param_size / (1024 ** 2):.2f} MB (parameters), {buffer_size /(1024 ** 2):.2f} MB (buffers)")
        wandb.log({"model_size_reduction":final_size/init_size}) 

    idx_vis = VARIABLES.index(field) # also prints out metrics for this field

    data = h5py.File(data_file, 'r')['fields'][ic:(ic+prediction_length*dt):dt,in_channels,0:img_shape_x]
    data = (data - means)/stds # standardize the data

    ensemble_init = make_ensemble_data(data, ensemble_size)
    ensemble_time = 0
    for i in range(ensemble_size):
        data_slice = ensemble_init[i] 

        data_slice = torch.as_tensor(data_slice).to(device, dtype=torch.float) # move to gpu for inference
        total_time, avg_time, acc_cpu, rmse_cpu, predictions_cpu, targets_cpu = inference(data_slice, model, prediction_length, idx=idx_vis,
                                                                                    params = params, device = device, 
                                                                                    img_shape_x = img_shape_x, img_shape_y = img_shape_y, std = std, 
                                                                                    m =m, field = field)
        ensemble_time += total_time
    print(f"tot ensemble time: {np.round(ensemble_time, 4)}s over {ensemble_size} members")

if __name__ == "__main__":
    wandb.login()
    run = wandb.init(
        project="weather-forecast-inference",  # Specify your project
    )
    main()

