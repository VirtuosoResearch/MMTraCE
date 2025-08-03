# Learning Multimodal Embeddings for Traffic Accident Prediction and Causal Estimation

<p align="center">
<a href="">Project Page</a> |
<a href="https://github.com/ziniuzhang/mmtrace/blob/main/static/MMTraCE_KDD_submission.pdf">Paper</a> |
<a href="https://huggingface.co/datasets/MichaelZona/MMTraCE">Dataset</a>

- Authors: [Ziniu Zhang](https://ziniuzhang.github.io/), [Minxuan Duan](https://www.minxuanduan.com/), [Haris Koutsopoulos](https://mobility.mit.edu/people/haris-koutsopoulos/) and [Hongyang R. Zhang](https://www.hongyangzhang.com/)

## Overview

This code implements **MMTraCE**, an multimodal learning framework for traffic accident prediction and causal estimation. 

We construct a large-scale dataset across $6$ U.S. states, comprising $9$ million traffic accident records, $1$ million high-resolution satellite images, and node-level structured features such as weather statistics, traffic volume, and road attributes. Each node in the road network is paired with a satellite image and associated temporal features, enabling localized multimodal learning. 

We propose a modeling framework that integrates visual encoders with graph neural network features. We implement three fusion mechanisms: a multilayer perceptron concatenation, a gated fusion network that adaptively balances visual and structural information, and a mixture of experts method to learn the features from different perspectives.

## Installation

### Prerequisites
To build up the environment, please run the following commands.

```bash
conda create -n traffic python=3.8
conda activate traffic
pip3 install torch torchvision torchaudio
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html # Please check the correct version of PyG
pip install -U scikit-learn
pip install torch_geometric
pip install pandas
pip install pyDataverse
pip install torch-geometric-temporal
pip install networkx karateclub
mkdir ./data
```

If there is an error about `to_dense_adj` due to the version confliction, please check this [link](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/issues/267).

## MMTraCE Dataset Structure
The dataset should be in the following structure (Use Delaware as an example):

```bash
./data/
└── DE/
    └── Road_Network_Edges_DE.csv
    └── Road_Network_Nodes_DE.csv # The road network of Delaware.
    └── accidents_monthly.csv # The accident record.
    └── adj_matrix.pt
    └── Edges/ # The dynamic and static feature of edges.
        └── edge_features_2009.pt
        ......
        └── edge_features_2024.pt
        └── edge_features.pt
    └── Nodes/ # The dynamic feature of nodes.
        └── node_features_2009_1.csv
        ......
        └── node_features_2024_10.csv

./embeddings/ # embeddings of satellite images.
└── vit_image/
    └── DE.npy
    ......
    └── NV.npy
└── clip_image/
    └── DE.npy
    ......
    └── NV.npy
```

Our dataset is located in [Huggingface](https://huggingface.co/datasets/MichaelZona/MMTraCE). You can also find it on Harvard datset.

## Usage

Here we report a sample script of running mixture-of-experts in Delaware for the classification task.

```bash
python train.py --state_name DE --node_feature_type node2vec\
    --encoder moe --num_gnn_layers 2 \
    --epochs 30 --lr 0.001 --runs 3 \
    --load_dynamic_node_features\
    --load_static_edge_features\
    --load_dynamic_edge_features\
    --train_years 2009 2010 2011 2012 2013\
    --valid_years 2014 2015 2016 2017 2018\
    --test_years  2019 2020 2021 2022 2023 2024\
    --device 0
```

The `--encoder` can be changed into `gnn_vision_mlp` and `gated_fusion` for the basic fusion and gated fusion methods. `--gnn_type` is for the type of basic gnn structure in the fusion method. `--vision_type` is for the type of image embedding using in the framework.

The encoder can also be changed into `gcn`, `gin`, `dcrnn`, and `graphormer` for GNN embedding, and `mlp` for simple multilayer perceptron.

For the regression task, please add `--train_accident_regression` in the script. For supervised contrastive learning, please add `--train_supcon` in the script.

We also list scripts to generate different node embeddings in `./embeddings`

---

Feel free to contact [zhang.zini@northeastern.edu](mailto:zhang.zini@northeastern.edu) if you have any questions.