from diffusers import DiffusionPipeline
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from datasets import load_dataset
import torch
import time
import os
import fire
import sys
import numpy as np

# A modified version of the inference script from project_utils.py
# Adapted for distributed learning across GPUs



def inference_ensemble(ensemble_init, model, prediction_length, idx, params, device, img_shape_x, img_shape_y, std, m, field):
    # Loop over ensemble idx and send to different GPUs
    # Specify ensemble_size with --num_processes flag passed through terminal

    ensemble_size = int(sys.argv[-1])

    # Use Accelerator() for distribution
    accelerator = Accelerator()
    device = accelerator.device

    # Prepare model with Accelerator
    model = accelerator.prepare(model)
    # Distribute ensemble indices
    local_ensemble_size = (ensemble_size + accelerator.num_processes - 1) // accelerator.num_processes
    start_idx = accelerator.process_index * local_ensemble_size
    end_idx = min(start_idx + local_ensemble_size, ensemble_size)


    ens_idx_results = []
    for ens in range(start_idx, end_idx):
        
        data_slice = ensemble_init[ens] 
       
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
                
                if accelerator.is_main_process: # Only write to wandb if we're in the main process
                    print('Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, field, rmse[i,idx], acc[i,idx]))
                    wandb.log({"accuracy": acc[i,idx], "rmse": rmse[i,idx], "step_time": iter_time})
                
                pred = future_pred
                tar = future
                total_time += iter_time

        if accelerator.is_main_process: # Only write to wandb if we're in the main process
            print(f'Total inference time: {total_time:.2f}s, Average time per step: {total_time/prediction_length:.2f}s')
            wandb.log({"total_inference_time": total_time, "avg_step_time": total_time/prediction_length})
        
        # copy to cpu for plotting and visualization
        ens_idx_results.append({
            "acc": acc_cpu,
            "rmse": rmse_cpu,
            "predictions": predictions_cpu,
            "targets": targets_cpu,
            "ensemble_idx": int(ens),
        })

        #Gather results across processes
        all_results = gather_object(results)
        if accelerator.is_main_process: # Only return gathered results if we're in the main process
            return gathered
        else:
            return None
