# Learning Multimodal Embeddings for Traffic Accident Prediction and Causal Estimation

- Authors: [Ziniu Zhang](https://ziniuzhang.github.io/), [Minxuan Duan](https://www.minxuanduan.com/), [Haris Koutsopoulos](https://mobility.mit.edu/people/haris-koutsopoulos/) and [Hongyang R. Zhang](https://www.hongyangzhang.com/)

## Overview

This code implements an multimodal learning framework for traffic accident prediction and causal estimation. We construct a large-scale dataset across $6$ U.S. states, comprising $9$ million traffic accident records, $1$ million high-resolution satellite images, and node-level structured features such as weather statistics, traffic volume, and road attributes. Each node in the road network is paired with a satellite image and associated temporal features, enabling localized multimodal learning. We propose a modeling framework that integrates visual encoders with graph neural network features. We implement three fusion mechanisms: a multilayer perceptron concatenation, a gated fusion network that adaptively balances visual and structural information, and a mixture of experts method to learn the features from different perspectives. Our approach achieves consistent improvements across states, with an $90.1\%$ average AUROC.

## Requirements

To build up the environment, please run the following commands.

```bash
conda create -n traffic python=3.8
conda activate traffic
pip3 install torch torchvision torchaudio
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install -U scikit-learn
pip install torch_geometric
pip install pandas
pip install pyDataverse
pip install torch-geometric-temporal
pip install networkx karateclub
```

If there is an error about `to_dense_adj` due to the version confliction, please check this [link](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/issues/267).
