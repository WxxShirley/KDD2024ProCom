# KDD2024Submission - ProCom

This is the official repo for KDD 2024 Research Track Submission paper ''**ProCom: A Few-shot Targeted Community Detection Algorithm**''.



## Run the Codes

### Repo Introduction
This repo contains the following contents:
```
.
├── README.md
├── ckpts               --- This folder contains pre-trained models for reproducibility 
├── code                --- This folder contains codes of ProCom
│   ├── data.py             (data loading)
│   ├── metrics.py          (evaluation)
│   ├── model.py            (GNN encoder and Prompt layer)
│   ├── pretrain.py         (context-aware pre-training)
│   ├── run.py              (run ProCom pipeline)
│   ├── run_pretrain.py     (run ProCom's pretraining phase)
│   └── utils.py            (utilization functions)
├── data                ---  This folder contaisn 5 experimental datasets
│   ├── amazon
│   ├── dblp
│   ├── facebook
│   ├── lj
│   └── twitter
└── logs                ---  This folder contains several running logs for references
    ├── AMAZON_EXAMPLE.log
    ├── DBLP_EXAMPLE.log
    └── FACEBOOK_EXAMPLE.log
```

### Environmental Requirements

0. Python 3.7 or above
1. Install pytorch with version 1.13.0 or later 
2. Install Pytorch-Geometric (PyG) with version 2.3.1. Please refer to [PyG official website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for more information of installing prerequisites.


### Running Scripts

Execute `code/run_pretrain.py` for the pre-training stage and saving the pre-trained model:
```
python run_pretrain.py --dataset=DATASET_NAME 
```

Execute `code/run.py` for the overall ProCom pipeline
```
python run.py --dataset=DATASET_NAME  --run_times=YOUR_RUN_TIMES --subg_scale=PARAMETER_LAMBDA
```

Descriptions of arguments (for more options, please refer to `run.py`)
```
--dataset [facebook, amazon, dblp, twitter, lj]: the dataset to run 
--num_shot: number of prompt communities, default as 10
--num_pred: number of predicted communities
--gnn_type [GCN, GAT, SAGE, GIN, TransformerConv]: gnn encoder type, default as GCN
--hidden_dim: embedding dimention, default as 128
--node_scale: weight of L_{n2c}, default as 1.0
--subg_scale: weight of L_{c2c}, default as 0.1, search fron {0.001, 0.01, 0.1, 1}
```

### Example Logs

We have uploaded some running logs under the `logs` folder for reference.




## Performance 



We conduct the overall performance comparison with both traditional community detection methods (BigClam, ComE, CommunityGAN) and semi-supervised methods (Bespoke, SEAL, CLARE).

For traditional methods, the results are reported from SEAL and CLARE. 
For semi-supervised methods, we run their released codes under 10-shot setting, and parameters are set following the original papers. 

* **Bespoke**: Semi-Supervised Community Detection Using Structure and Size. In *ICDM*, 2018.  [Official Implementation](https://github.com/abaxi/bespoke-icdm18) 
* **SEAL**: SEAL: Learning Heuristics for Community Detection with Generative Adversarial Networks. In *KDD*, 2020. [Official Implementation](https://github.com/yzhang1918/kdd2020seal)
* **CLARE**: CLARE: A Semi-supervised Community Detection Algorithm. In *KDD*, 2022. [Official Implementation](https://github.com/FDUDSDE/KDD2022CLARE)
  
| F1-Score     | $\text{BigClam}$ | $\text{ComE}$  | $\text{CommunityGAN}$ | $\text{Bespoke}$            | $\text{SEAL}$               | $\text{CLARE}$              | $\text{ProCom (Ours)}$               |
| ----------- | ------- | ----- | ------------ | ------------------ | ------------------ | ------------------ | --------------------------- |
| Facebook    | $32.92$   | $27.92$ | $32.05$        | $29.67_{\pm 0.85}$ | $31.10_{\pm 3.84}$ | $28.53_{\pm 1.36}$ | $\textbf{38.57}_{\pm 2.02}$ |
| Amazon      | $53.79$   | $48.23$ | $51.09$        | $80.38_{\pm 0.64}$ | $82.26_{\pm 4.04}$ | $78.89_{\pm 2.10}$ | $\textbf{84.36}_{\pm 0.23}$ |
| Livejournal | $39.17$   | $\text{N/A}$  | $40.67$        | $30.98_{\pm 1.55}$ | $42.85_{\pm 2.60}$ | $45.38_{\pm 4.07}$ | $\textbf{54.35}_{\pm 3.04}$ |
| DBLP        | $40.41$   | $25.24$ | $\text{N/A}$          | $41.55_{\pm 0.40}$ | $41.74_{\pm 6.35}$ | $48.75_{\pm 2.51}$ | $\textbf{50.96}_{\pm 1.57}$ |
| Twitter     | $24.33$   | $15.89$ | $\text{N/A}$          | $29.85_{\pm 0.15}$ | $16.97_{\pm 1.32}$ | $20.05_{\pm 0.88}$ | $\textbf{31.09}_{\pm 0.35}$ |



