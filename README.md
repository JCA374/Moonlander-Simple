# Simple MoonLander AI

A lightweight Deep Q-Network (DQN) implementation optimized for CPU training on the LunarLander-v2 environment.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py
```
Training typically takes 500-1000 episodes. The model saves automatically when it reaches a score of 200+.

### Evaluation
```bash
python evaluate.py
```
Loads the trained model and runs 10 evaluation episodes with visualization.

## Optimizations for Your Hardware

- Small neural network (64-64 hidden layers)
- CPU-only training 
- Limited memory buffer (10,000 experiences)
- Efficient batch processing (32 samples)
- Target network updates every 100 episodes

Expected training time: 10-20 minutes on Intel i7-8565U.