# Continual-Mega: A Large-scale Benchmark for Generalizable Continual Anomaly Detection

This repository contains the training and evaluation code for the Continual-Mega benchmark. \
We provide the checkpoint files of the proposed model along with the training and evaluation code for the Continual-Mega benchmark.

## Requirements
```
pip install -r requirements.txt
```

## Datasets
### Download
The datasets are available through Hugging Face. \
https://huggingface.co/datasets/Continual-Mega/Continual-MEGA-Benchmark

### Structure
The datasets directory should be structured as follows:
``` 
data/
├── continual_ad/              # Our proposed ContinualAD dataset
├── mvtec_anomaly_detection/   # MVTec-AD dataset
├── VisA_20220922/             # VisA dataset
├── VIADUCT/                   # VIADUCT dataset 
├── Real-IAD-512/              # RealIAD dataset (512 size)
├── MPDD/                      # MPDD dataset
└── BTAD/                      # BTAD

``` 

## CLIP Pretrained Weights
Download Link: https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt

Please download the CLIP pretrained weights file from the link above and place it in the `CLIP/ckpt` directory.

## Training
### Base Classes
```
sh train_scripts/train_base_classes.sh
```

### Continual Learning - Scenario 1
```
sh train_scripts/scenario1_continual_5classes.sh # 5 Classes
sh train_scripts/scenario1_continual_10classes.sh # 10 Classes
sh train_scripts/scenario1_continual_30classes.sh # 30 Classes
```

### Continual Learning - Scenario 2
```
sh train_scripts/scenario2_continual_5classes.sh # 5 Classes
sh train_scripts/scenario2_continual_10classes.sh # 10 Classes
sh train_scripts/scenario2_continual_30classes.sh # 30 Classes
```

### Continual Learning - Scenario 3
```
sh train_scripts/scenario3_continual_5classes.sh # 5 Classes
sh train_scripts/scenario3_continual_10classes.sh # 10 Classes
sh train_scripts/scenario3_continual_30classes.sh # 30 Classes
```

## Evaluation
### checkpoint files
We provide checkpoint files for the setting with 30 classes per task in Scenario 2.

### Continual Settings
``` 
sh eval_continual.sh
``` 
### Zero-Shot Generalization
``` 
sh eval_zero.sh
``` 
