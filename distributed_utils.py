import torch
import time

# import modules for distributed inference utilities across GPUs
from accelerate import PartialState
from diffusers import DiffusionPipeline

def inference_ensemble(initial_condition, model, prediction_length, device):

    ensemble_size = initial_condition.shape[0]
    predictions = [initial_condition]  # store the initial condition as the first "prediction"
    
    # synchronize before timing if using GPU
    if device != 'cpu':
        torch.cuda.synchronize()
    start_time = time.time()
    
    # autoregressive loop
    # adapted for multi-GPU support
    # Loop over batches to get initial conditions
    # use distributed_state.split_between_processes()
    with torch.no_grad():
        current = initial_condition
        for step in range(1, prediction_length):
            # compute model prediction for the next step
            output = model(current)
            predictions.append(output)
            # update current state for next autoregressive step
            current = output
    
    # synchronize after inference to ensure accurate timing
    if device != 'cpu':
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    print(f"Total inference time for {ensemble_size} ensemble members over {prediction_length} steps: {total_time:.3f} seconds")
    return predictions, total_time
