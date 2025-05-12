from accelerate import Accelerator
import os, sys
sys.path.insert(1, './FourCastNet/') # insert code repo into path
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import click
import wandb


from constants import VARIABLES
from proj_utils import make_ensemble_data, inference, load_data_and_model
from quantize import replace_linear_with_target_and_quantize, W8A16LinearLayer, model_size
from distributed_inference import inference_ensemble

from utils.YParams import YParams
os.environ["ACCELERATE_USE_TP"] = "false"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
config_file = "./FourCastNet/config/AFNO.yaml"
config_name = "afno_backbone"
img_shape_x = 720
img_shape_y = 1440

QDTYPE = torch.int8

@click.command()
@click.option("--compile", is_flag=True, help="torch.compile()")
@click.option(
    "--quantize", is_flag=True, help="quantize)"
)
@click.option(
    "--distributed", is_flag=True, help="distributed)"
)
@click.option(
    "--ensemble_size",
    type=int,
    default = 1,
    help="how many forecast members?",
)
@click.option(
    "--prediction_length",
    type=int,
    default = 20,
    help="how many time steps to run? each 6 hrs",
)
@click.option(
    "--field",
    default="u10",
    type=click.Choice([
        'u10', 'v10', 't2m', 'sp', 'msl', 't850', 
        'u1000', 'v1000', 'z1000', 'u850', 'v850', 'z850',
        'u500', 'v500', 'z500', 't500', 'z50', 'r500', 'r850', 'tcwv'
    ]),
    help="""Field of interest:
    u10: 10 metre zonal wind speed (m s-1)
    v10: 10 metre meridional wind speed (m s-1)
    t2m: 2 metre temperature (K)
    sp: Surface pressure (Pa)
    msl: Mean sea level pressure (Pa)
    t850: Temperature at 850 hPa (K)
    u1000/v1000/z1000: Wind components at 1000 mbar (m s-1)
    u850/v850/z850: Wind components at 850 mbar (m s-1)
    u500/v500/z500: Wind components at 500 mbar (m s-1)
    t500: Temperature at 500 mbar (K)
    z50: Geopotential height at 50 hPa
    r500: Relative humidity at 500 mbar
    r850: Relative humidity at 850 mbar
    tcwv: Total column water vapor (kg m-2)
    """,
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
def main(compile, quantize, distributed, ensemble_size, prediction_length, field, base_path, file_path):

    run = wandb.init(
        project="weather-forecast-inference", entity="jhalpern-columbia-university",   # Specify your project
        config={                         # Track hyperparameters and metadata
                "quantize": quantize,
                "compile": compile, 
                "qdtype": QDTYPE,
                "ensemble_size": ensemble_size,
                "variable": field,
                "distributed":distributed
        },
    )

    params = YParams(config_file, config_name)
    print("Model architecture used = {}".format(params["nettype"]))

    model, data_file, in_channels, device, params = load_data_and_model(
        base_path, params
    )

    if compile:
        model = torch.compile(model, backend="inductor")

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
    if not distributed:
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
        n_gpu = 1
    else:
        hold = data[np.newaxis, :, :, :]
        ensemble_init = np.tile(hold, (ensemble_size, 1, 1, 1, 1))

        epsilon = 1e-8
        random_values = np.random.uniform(0, 10, ensemble_size)

        for i in range(ensemble_size):
            ensemble_init[i, :, :, :] *= epsilon * random_values[i]

        # Run the ensemble inference and measure the performance
        all_results = inference_ensemble(ensemble_init, model, prediction_length, idx_vis, params, device = device,img_shape_x = img_shape_x, img_shape_y = img_shape_y, 
                                         std = std, m =m, field = field, ensemble_size = ensemble_size)
        ensemble_time = all_results[0]['total_inference_time_for_ensemble']
        n_gpu = all_results[0]['n_gpu']

    if file_path:
        if not distributed:
            df = pd.DataFrame(
                {
                    "tot_time": [ensemble_time],
                    "avg_time": [ensemble_time/ensemble_size],
                    "compiled": [compile],
                    "quantize": [quantize],
                    "distributed": [distributed],
                    "num_gpu":[n_gpu]
                }
            )

            if os.path.exists(file_path):
                df.to_csv(file_path, mode="a", header=False, index=False)
            else:
                df.to_csv(file_path, index=False)
        if distributed:
            accelerator = Accelerator()
            if accelerator.is_main_process:
                df = pd.DataFrame(
                    {
                        "tot_time": [ensemble_time],
                        "avg_time": [ensemble_time/ensemble_size],
                        "compiled": [compile],
                        "quantize": [quantize],
                        "distributed": [distributed],
                        "num_gpu":[n_gpu]
                    }
                )

                if os.path.exists(file_path):
                    df.to_csv(file_path, mode="a", header=False, index=False)
                else:
                    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    os.environ["WANDB_NOTEBOOK_NAME"] = './base_script.py'
    wandb.login()
    main()
