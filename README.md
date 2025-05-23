# HPML Project: [Accelerating Inference in State-of-the-Art Weather Forecast Models]

## Team Information
- **Team Name**: [Climate Crew]
- **Members**:
  - Amelia Chambliss (ac5114)
  - Jake Halpern (jmh2363)
  - Maggie Powell (mp4257)


## 1. Problem Statement:
Numerical weather prediction is used to forecast environmental conditions from present to several weeks into the future. It is a computationally expensive and challenging task, with prediction skill decreasing with time. Traditionally, physics-based numerical models are used for this task and require hundreds of hours on a supercomputer to produce a single forecast. Deep learning has provided major advances in recent years, with ML-based models from Google [2], NVIDIA [3], Huawei [1], and others outperforming traditional models in terms of forecast accuracy. Once trained, these models can produce a forecast in seconds. 

The objective of the final project is to optimize and distribute inference of NVIDIA's FourCastNet numerical weather prediction model. While a single forecast provides useful information, it is highly uncertain due to sensitively of the model to initial conditions. Typically, ensembles of 50 to 100 members with perturbed initial conditions are used to produce forecasts. Efficient inference would allow modeling centers to produce even larger ensembles of ML-based weather forecasts, thereby providing a more robust estimate of mean conditions and extreme events. For this project we will examine how quantization, compilation of the computational graph, and distributed training impacts inference performance. When experimenting with quantization, we will also carefully examine the effect on model accuracy and skill. 

[1] Kaifeng Bi et al. “Pangu-weather: A 3d high-resolution model for fast and accurate global weather forecast”. In: arXiv preprint arXiv:2211.02556 (2022).
[2] Remi Lam et al. “Learning skillful medium-range global weather forecasting”. In: Science 382.6677 (2023), pp. 1416–1421.
[3] Jaideep Pathak et al. “Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators”. In: arXiv preprint arXiv:2202.11214 (2022).


## 2. Model Description
Summarize the model architecture(s) used (e.g., ResNet-18, Transformer). Include:
- Framework (e.g., PyTorch, TensorFlow)
- Any custom layers or changes to standard models


base script.py processes user inputs from the command line and implements the corresponding test. It does so by calling the inference functions defined in proj_utils.py (for one GPU) and distributed_inference.py (for multi-GPU). Quantization is implemented by packages from quantize.py. It loads the model and computes accuracies and root mean square errors for each timestep using calls to functions defined in proj_utils.py. A schematic of the code structure is shown below:


![code-flow](https://github.com/user-attachments/assets/e352c08e-3422-4bed-b307-d5e73e36eda5)


---

## 3. Final Results Summary

1) Torch.compile() speedup:
<img width="1078" alt="Screen Shot 2025-05-08 at 9 28 34 PM" src="https://github.com/user-attachments/assets/2389a050-c7ff-4ce4-a6b0-2e2feb0dcb4c" />
We observed a 30% speedup after using the inductor backend for torch.compile.

2) Quantization model size reduction: 

<img width="499" alt="Screen Shot 2025-05-08 at 9 28 23 PM" src="https://github.com/user-attachments/assets/c6cbb3d9-60e4-4994-8c1d-01b389ff804b" />
We observed a 58% model size reduction after weight quantization with int8.

3) Distributed Inference speedups:
 <img width="475" alt="Screen Shot 2025-05-08 at 9 28 42 PM" src="https://github.com/user-attachments/assets/23b757b8-3c6f-4413-a513-cc2d0099d341" />
We observed a 28% speedup from 1 to 4 GPUs.

4) Compined optimization:

<img width="476" alt="Screen Shot 2025-05-08 at 9 28 48 PM" src="https://github.com/user-attachments/assets/335e8fbf-bf5d-4b2b-a95c-6339b2b26aac" />

After combining all of these optimizations, we achieved a 44% speedup in inference time.



## 4. Reproducibility Instructions 

Implementation of inference tests is in **base_script.py**. This script can be used to do the inference using FourCastNet and the wrapper functions defined in this repository. This repo contains support for quantized weights to reduce model size, torch.compile to reduce inference times, and distributed inference to improve speedup. First, clone the github repo to your favorite computing environment with GPU support. To run this script, follow the steps below: 

### A. Requirements

Install conda environment from the .yml file 
```   
     conda env create -f environment.yml
     conda activate hpml_env 
```
 Install hugging face packages for multi-gpu: 
 ```  
        pip install accelerate 
 ```  
Install the ccai demo file in your chosen directory with the following lines: 
```
     wget https://portal.nersc.gov/project/m4134/ccai_demo.tar 
   
     tar -xvf ccai_demo.tar 
   
     rm ccai_demo.tar 
```   
pass  ``` --base_path <PATH_TO_YOUR_DATA_FROM_STEP_2>   ```  to base_script.py with the path to the data from ccai_demo.tar

---

B. Wandb Dashboard

You can find our experiments and data in the Weights and Biases team here: https://wandb.ai/jhalpern-columbia-university/weather-forecast-inference 

---

### C. Specify for Training or For Inference or if Both 

The project is focused on inference only. To compare the baseline run to the run with all optimizations run:


 ```  
     python base_script.py --ensemble_size 10 
 ``` 

 and

  ```  
     accelerate launch --multi_gpu --num_processes=4 base_script.py --distributed --ensemble_size 10 --compile --quantize
 ``` 

### D. Evaluation

Determine desired parameters for testing. The options are:

     --**compile**: This switch will turn on usage of torch.compile, which pre-compiles the code to enable speedup.
   
     --**qunatize**: This switch will turn on int8 quantization of the linear layer model weights, reducing the size of the model overall.
   
     --**distributed**: This switch determines whether you're running the inference across multiple GPUs or not. Doing so can reduce runtimes. This uses the Hugging Face Accelerate package: https://huggingface.co/docs/diffusers/en/training/distributed_inference
   
     --**prediction_length**: This parameter controls the number of timesteps for the autoregressive loop used during inference. Each timestep corresponds to 6 hours of advancement in weather behavior predicted by FourCastNet
   
     --**ensemble_size**: Size of ensemble you want to use. The ensemble is the set of randomly perturbed inputs fed to the inference model. These inputs are perturbed from real weather forecast data in the ERA5 dataset

     --**field**: String, variable name you'd like to calculate. The full list of weather variables is provided at the top of the base_script.py file. These include quantities such as wind speeds and temperatures. A list of these variables used by dependent scripts in this repo can be found in constants.py.

For more information simply run:

```
python base_script.py --help
```

Run base script with chosen flags. 

For a single GPU, you can run the base script in the following way:
   
     python base_script.py --ensemble_size 10
     python base_script.py --ensemble_size 10 --quantize --compile
   

For a distributed inference use case, you will need to run using arguments for hugging face accelerate. Here's an example for 4 gpus:
   
     accelerate launch --multi_gpu --num_processes=4 base_script.py --distributed --ensemble_size 10

Alternatively, a Jupyter notebook implementation is available in the **base_notebook.ipynb** file provided. Note that if running base_script.py, every input variable must be specified. 

The torch_compile_test.py script was used to compare various torch.compile() settings and saves 
benchmark info to a csv. Info on its arguments can be found with 

     python torch_compile.py --help

Plots for torch compile testing and quantization are in torch_compile_plots.ipynb and plot_quantization.py respectively.

---

### E. Quickstart: Minimum Reproducible Result

```bash
# Step 1: Set up environment
conda env create -f environment.yml 
conda activate hpml_env
pip install accelerate

# Step 2: Download dataset
wget https://portal.nersc.gov/project/m4134/ccai_demo.tar 
tar -xvf ccai_demo.tar 
rm ccai_demo.tar 

# Step 3: Inference
python base_scripy.py --base_path <PATH_TO_YOUR_DATA_FROM_STEP_2>
```

---


## Notes-- Understanding outputs:

Outputs for each run will be printed to stdout. This includes total runtime for inference as well as average inference time per ensemble member, accuracy, and root mean square errors. These can be visualized in WandB to evaluate performance across multiple tests. 

As additional functionality, we've provided the option to pass ```--file_path <NAME_OF_CSV>.csv``` to base_script.py to save total ensemble time and the run configuration to a csv.

Outputs for quantization test can be found in **quantization_tests.csv** and are plotted in **quantization_results.png**. 

Torch.compile test results are saved in **torch_compile.csv** and plotted in **torch_compile_tests.ipynb**. Data for distributed inference on Perlmutter with NVIDIA A100 GPUs is saved in **distributed_inference.csv**. Torch.compile and quanitzation tests can be conducted using the corresponding .py files. 

