# Continual-MEGA: A Large-scale Benchmark for Generalizable Continual Anomaly Detection

Official implementation of **Continual-MEGA: A Large-scale Benchmark for Generalizable Continual Anomaly Detection**, published in *Neurocomputing*.

[[Paper](https://www.sciencedirect.com/science/article/pii/S0925231226018588)]
[[arXiv](https://arxiv.org/abs/2506.00956)]
[[Dataset](https://huggingface.co/datasets/Continual-Mega/Continual-MEGA-Benchmark)]

This repository provides the training and evaluation code for the Continual-MEGA benchmark, along with the checkpoints of our proposed model.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Datasets

### Download

The Continual-MEGA benchmark datasets are available through Hugging Face:

https://huggingface.co/datasets/Continual-Mega/Continual-MEGA-Benchmark

### Directory Structure

The dataset directory should be structured as follows:

```text
data/
├── continual_ad/              # Our proposed ContinualAD dataset
├── mvtec_anomaly_detection/   # MVTec AD dataset
├── VisA_20220922/             # VisA dataset
├── VIADUCT/                   # VIADUCT dataset
├── Real-IAD-512/              # Real-IAD dataset resized to 512
├── MPDD/                      # MPDD dataset
└── BTAD/                      # BTAD dataset
```

## CLIP Pretrained Weights

Download the pretrained CLIP ViT-L/14@336px weights from the following link:

https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt

Place the downloaded file in the following directory:

```text
CLIP/ckpt/ViT-L-14-336px.pt
```

## Training

### Base Classes

```bash
sh train_scripts/train_base_classes.sh
```

### Continual Learning — Scenario 1

```bash
sh train_scripts/scenario1_continual_5classes.sh   # 5 classes per task
sh train_scripts/scenario1_continual_10classes.sh  # 10 classes per task
sh train_scripts/scenario1_continual_30classes.sh  # 30 classes per task
```

### Continual Learning — Scenario 2

```bash
sh train_scripts/scenario2_continual_5classes.sh   # 5 classes per task
sh train_scripts/scenario2_continual_10classes.sh  # 10 classes per task
sh train_scripts/scenario2_continual_30classes.sh  # 30 classes per task
```

### Continual Learning — Scenario 3

```bash
sh train_scripts/scenario3_continual_5classes.sh   # 5 classes per task
sh train_scripts/scenario3_continual_10classes.sh  # 10 classes per task
sh train_scripts/scenario3_continual_30classes.sh  # 30 classes per task
```

## Evaluation

### Checkpoint Files

We provide checkpoint files for the Scenario 2 setting with 30 classes per task.

### Continual Learning Evaluation

```bash
sh eval_continual.sh
```

### Zero-Shot Generalization Evaluation

```bash
sh eval_zero.sh
```

## Citation

Please cite our paper if you use the Continual-MEGA benchmark, dataset, or code in your research:

```bibtex
@article{lee2026continual,
  title={Continual-MEGA: A large-scale benchmark for generalizable continual anomaly detection},
  author={Geonu Lee and Yujeong Oh and Geonhui Jang and Soyoung Lee and Jeonghyo Song and Sungmin Cha and YoungJoon Yoo},
  journal = {Neurocomputing},
  volume = {700},
  pages = {134460},
  year = {2026},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2026.134460},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231226018588},
}
```
