# HPML_TermProject
CS HPML Spring 2025 Term Project
Team members: Margaret Powell, Jacob Halpern, Amelia Chambliss

## Objective:
Numerical weather prediction is used to forecast environmental conditions from present to several weeks into the future. It is a computationally expensive and challenging task, with prediction skill decreasing with time. Traditionally, physics-based numerical models are used for this task and require hundreds of hours on a supercomputer to produce a single forecast. Deep learning has provided major advances in recent years, with ML-based models from Google [2], NVIDIA [3], Huawei [1], and others outperforming traditional models in terms of forecast accuracy. Once trained, these models can produce a forecast in seconds. 

The objective of the final project is to optimize and distribute inference of NVIDIA's FourCastNet numerical weather prediction model. While a single forecast provides useful information, it is highly uncertain due to sensitively of the model to initial conditions. Typically, ensembles of 50 to 100 members with perturbed initial conditions are used to produce forecasts. Efficient inference would allow modeling centers to produce even larger ensembles of ML-based weather forecasts, thereby providing a more robust estimate of mean conditions and extreme events. For this project we will examine how quantization, compilation of the computational graph, and distributed training impacts inference performance. When experimenting with quantization, we will also carefully examine the effect on model accuracy and skill. 

[1] Kaifeng Bi et al. “Pangu-weather: A 3d high-resolution model for fast and accurate global weather forecast”. In: arXiv preprint arXiv:2211.02556 (2022).
[2] Remi Lam et al. “Learning skillful medium-range global weather forecasting”. In: Science 382.6677 (2023), pp. 1416–1421.
[3] Jaideep Pathak et al. “Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators”. In: arXiv preprint arXiv:2202.11214 (2022).
