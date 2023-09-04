# Multi-modal Graph Convolutional Network with Composing Contrastive Augmentations for Disease Prediction

## About
-

## Prerequisites
- `Python 3.9.0`
- `Pytorch 1.12.1`
- `torch-geometric 2.2.0`
- `scikit-learn 0.24.2`
- `NumPy 1.23.5`

This code has been tested using `Pytorch` on a NVIDIA GeForce RTX 3090.

## Training
```
./scripts/train_ABIDE.sh
./scripts/train_ADNI.sh
./scripts/train_ODIR.sh
```
or run
```
python train_eval_evgcn.py --dataset ABIDE --type Degree --num_classes 2 --train 1
python train_eval_evgcn.py --dataset ADNI --type Degree --num_classes 2 --train 1
python train_eval_evgcn.py --dataset ODIR --type Degree --num_classes 8 --train 1
```
