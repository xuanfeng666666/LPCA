# LPCA:

![Markdown Logo](./framwork.png)

## Quick Start
### Environment variables & dependencies
```
torch                     2.2.0+cu118
torch_cluster             1.6.3+pt24cu118
torch-geometric           2.6.1
torch_scatter             2.1.2+pt24cu118
torch_sparse              0.6.18+pt24cu118
torch_spline_conv         1.2.2+pt24cu118
dgl                       2.2.0+cu118

conda create -n DSTAG python=3.8
conda activate DSTAG
```

### Process data

#### For the three datasets ICEWS18, ICEWS14 and ICEWS05-15, go into the dataset folder in the ./ directory and run the following command.
```
python new_dataprocess.py
```
### Train & Test models

#### 1. Train & Test models
```
python main_pre.py
```
