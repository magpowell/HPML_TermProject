import os, sys, time
import numpy as np
import h5py
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
from networks.afnonet import AFNONet

from collections import OrderedDict

def make_ensemble_data(data, ensemble_size):
    if ensemble_size == 1:
        return data[np.newaxis, :, :, :]
    
    hold = data[np.newaxis, :, :, :]
    ensemble_init = np.tile(hold, (ensemble_size, 1, 1, 1, 1))

    epsilon = 1e-8
    random_values = np.random.uniform(0, 10, ensemble_size)

    for i in range(ensemble_size):
        ensemble_init[i, :, :, :] *= epsilon * random_values[i]
    return ensemble_init


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

def inference(data_slice, model, prediction_length, idx, params, device, img_shape_x, img_shape_y, std, m, field):
    # torch.compile warmup
    with torch.no_grad():
        dummy_input = torch.randn(1, data_slice.shape[1], img_shape_x, img_shape_y).to(device)
        _ = model(dummy_input)

    # create memory for the different stats
    n_out_channels = params['N_out_channels']
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    rmse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)

    # to conserve GPU mem, only save one channel (can be changed if sufficient GPU mem or move to CPU)
    targets = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    predictions = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

    total_time = 0
    with torch.no_grad():
        for i in range(data_slice.shape[0]):
            iter_start = time.perf_counter()
            if i == 0:
                first = data_slice[0:1]
                future = data_slice[1:2]
                pred = first
                tar = first
                # also save out predictions for visualizing channel index idx
                targets[0,0] = first[0,idx]
                predictions[0,0] = first[0,idx]
                # predict
                future_pred = model(first)
            else:
                if i < prediction_length - 1:
                    future = data_slice[i+1:i+2]
                future_pred = model(future_pred) # autoregressive step

            if i < prediction_length - 1:
                predictions[i+1,0] = future_pred[0,idx]
                targets[i+1,0] = future[0,idx]

            # compute metrics using the ground truth ERA5 data as "true" predictions
            rmse[i] = weighted_rmse_channels(pred, tar) * std
            acc[i] = weighted_acc_channels(pred-m, tar-m)
            iter_end = time.perf_counter()
            iter_time = iter_end - iter_start
            print('Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, field, rmse[i,idx], acc[i,idx]))
            wandb.log({"accuracy": acc[i,idx], "rmse": rmse[i,idx], "step_time": iter_time})

            pred = future_pred
            tar = future
            total_time += iter_time
    avg_time = total_time/prediction_length
    print(f'Total inference time: {total_time:.2f}s, Average time per step: {avg_time:.2f}s')
    wandb.log({"total_inference_time": total_time, "avg_step_time":avg_time})

    # copy to cpu for plotting/vis
    acc_cpu = acc.cpu().numpy()
    rmse_cpu = rmse.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    targets_cpu = targets.cpu().numpy()

    return total_time, avg_time, acc_cpu, rmse_cpu, predictions_cpu, targets_cpu

def load_model(model, params, checkpoint_file):
    ''' helper function to load model weights '''
    load_time_start = time.perf_counter()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname, weights_only=False)
    try:
        ''' FourCastNet is trained with distributed data parallel
            (DDP) which prepends 'module' to all keys. Non-DDP
            models need to strip this prefix '''
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval() # set to inference mode
    load_time_end = time.perf_counter()
    load_time = load_time_end - load_time_start
    print(f"Load time: {load_time} seconds")
    return model

def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

def latitude_weighting_factor(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

def weighted_rmse_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

def weighted_acc_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
    target, dim=(-1,-2)))
    return result
