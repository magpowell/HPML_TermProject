import os
import sys
import time

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torch._inductor.config 
import wandb

sys.path.insert(1, "./FourCastNet/")  # insert code repo into path

from networks.afnonet import AFNONet
from utils.YParams import YParams

from constants import VARIABLES
from distributed_utils import inference_ensemble
from proj_utils import (inference, lat, latitude_weighting_factor, load_model,
                        weighted_rmse_channels)

config_file = "./FourCastNet/config/AFNO.yaml"
config_name = "afno_backbone"
os.environ[
    "WANDB_NOTEBOOK_NAME"
] = "./torch_compile_test.py"  # this will be the name of the notebook in the wandb project database


def load_data_and_model(base_path, params):
    # data and model paths
    data_path = f"{base_path}ccai_demo/data/FCN_ERA5_data_v0/out_of_sample"
    data_file = os.path.join(data_path, "2018.h5")
    model_path = f"{base_path}ccai_demo/model_weights/FCN_weights_v0/backbone.ckpt"
    global_means_path = f"{base_path}ccai_demo/additional/stats_v0/global_means.npy"
    global_stds_path = f"{base_path}ccai_demo/additional/stats_v0/global_stds.npy"
    time_means_path = f"{base_path}ccai_demo/additional/stats_v0/time_means.npy"

    # import model
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    # in and out channels: FourCastNet uses 20 input channels corresponding to 20 prognostic variables
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    params["N_in_channels"] = len(in_channels)
    params["N_out_channels"] = len(out_channels)
    params.means = np.load(global_means_path)[
        0, out_channels
    ]  # for normalizing data with precomputed train stats
    params.stds = np.load(global_stds_path)[0, out_channels]
    params.time_means = np.load(time_means_path)[0, out_channels]

    # load the model
    if params.nettype == "afno":
        model = AFNONet(params).to(device)  # AFNO model
    else:
        raise Exception("not implemented")
    # load saved model weights
    model = load_model(model, params, model_path)
    model = model.to(device)
    return model, data_file, in_channels, device, params


@click.command()
@click.option("--use_compile", is_flag=True, help="torch.compile()")
@click.option(
    "--reduce_overhead", is_flag=True, help="reduce_overhead for torch.compile()"
)
@click.option("--eager", is_flag=True, help="eager backend for torch.compile()")
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
def main(use_compile, reduce_overhead, eager, base_path, file_path):
    # default
    params = YParams(config_file, config_name)
    print("Model architecture used = {}".format(params["nettype"]))

    model, data_file, in_channels, device, params = load_data_and_model(
        base_path, params
    )

    if use_compile:
        if reduce_overhead:
            model = torch.compile(model, backend="inductor", mode="reduce-overhead")
        elif eager:
            model = torch.compile(model, backend="eager")
        else:
            model = torch.compile(model, backend="inductor")

    # Ensure CUDA Graphs' tensor outputs are not overwritten
    if use_compile:
        torch.compiler.cudagraph_mark_step_begin()

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
    prediction_length = 20  # number of steps (x 6 hours)

    # which field to track for visualization
    field = "u10"
    idx_vis = VARIABLES.index(field)  # also prints out metrics for this field

    # get prediction length slice from the data
    print("Loading inference data")
    print("Inference data from {}".format(data_file))
    data = h5py.File(data_file, "r")["fields"][
        ic : (ic + prediction_length * dt) : dt, in_channels, 0:img_shape_x
    ]
    print(data.shape)
    print("Shape of data = {}".format(data.shape))

    # run inference
    data = (data - means) / stds  # standardize the data
    data = torch.as_tensor(data).to(
        device, dtype=torch.float
    )  # move to gpu for inference
    total_time, avg_time, acc, rmse, predictions, targets = inference(
        data,
        model,
        prediction_length,
        idx=idx_vis,
        params=params,
        device=device,
        img_shape_x=img_shape_x,
        img_shape_y=img_shape_y,
        std=std,
        m=m,
        field=field,
    )

    if file_path:
        df = pd.DataFrame(
            {
                "tot_time": [total_time],
                "avg_time": [avg_time],
                "avg_accuracy": [np.mean(acc)],
                "avg_rmse": [np.mean(rmse)],
                "compiled": [use_compile],
                "eager": [eager],
                "reduce_overhead": [reduce_overhead],
            }
        )

        if os.path.exists(file_path):
            df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            df.to_csv(file_path, index=False)


if __name__ == "__main__":
    wandb.login()
    run = wandb.init(
        project="weather-forecast-inference",  # Specify your project
    )

    main()
