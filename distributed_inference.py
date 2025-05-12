from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
import torch
import time

from proj_utils import lat, latitude_weighting_factor

# A modified version of the inference script from project_utils.py
# Adapted for distributed learning across GPUs

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


def inference_ensemble(ensemble_init, model, prediction_length, idx, params, device, img_shape_x, img_shape_y, std, m, field, ensemble_size):
    # Loop over ensemble idx and send to different GPUs

    # Use Accelerator() for distribution
    accelerator = Accelerator()
    device = accelerator.device
    n_gpu = accelerator.num_processes
    # Prepare model with Accelerator
    model = accelerator.prepare(model)
    # Distribute ensemble indices
    local_ensemble_size = (ensemble_size + accelerator.num_processes - 1) // accelerator.num_processes
    start_idx = accelerator.process_index * local_ensemble_size
    end_idx = min(start_idx + local_ensemble_size, ensemble_size)

    ens_idx_results = []
    for ens in range(start_idx, end_idx):
        
        data_slice = ensemble_init[ens] 
        data_slice = torch.tensor(data_slice, device=device, dtype=torch.float)
        
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

        total_time_elapsed = 0
        total_time_ensemble=0
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
                rmse[i] = weighted_rmse_channels(pred, tar) * std.to(pred.device)
                acc[i] = weighted_acc_channels(pred - m.to(pred.device), tar - m.to(pred.device))
                iter_end = time.perf_counter()
                iter_time = iter_end - iter_start
                
                pred = future_pred
                tar = future
                total_time_ensemble += iter_time

        
        # copy to cpu for plotting and visualization
        ens_idx_results.append({
            "total_inference_time_for_ensemble": total_time_ensemble,
            "ensemble_idx": ens,
            "n_gpu": n_gpu,
        })
        total_time_elapsed += total_time_ensemble

    #Gather results across processes
    all_results = gather_object(ens_idx_results)

    if accelerator.is_main_process:
        #total_time = sum(res["total_inference_time"] for res in all_results)
        avg_time = total_time_elapsed / ensemble_size

        print(f"\nTotal elapsed inference time across {ensemble_size} ensembles: {total_time_elapsed:.4f} seconds")
        print(f"Average time per ensemble member: {avg_time:.4f} seconds")

    return all_results
