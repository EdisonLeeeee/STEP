# STEP: Self-supervised Temporal Graph Pruning at Scale

PyTorch implementation of the paper "STEP: Self-supervised Temporal Graph Pruning at Scale".

# Requirements
+ torch == 1.8.1
- pytorch-lightning == 1.6.4
- torch_scatter == 2.0.8
- scikit-learn == 1.0.2
- scipy == 1.7.3


## Preprocessing

### Dataset
Create a folder 'dataset' to store data file.

[Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)  
[Reddit](http://snap.stanford.edu/jodie/reddit.csv)  

### Preprocess the data
We use the data processing method of the reference [TGAT](https://openreview.net/pdf?id=rJeW1yHYwH), [repo](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs#inductive-representation-learning-on-temporal-graphs-iclr-2020).

We use the dense npy format to save the features in binary format. If edge features or nodes features are absent, it will be replaced by a vector of zeros.   
  
    python build_dataset_graph.py --data wikipedia --bipartite
    python build_dataset_graph.py --data reddit --bipartite

## Model Training
Training the Graph pruning network based on an unsupervised task.

    python train_gsn.py --data_set wikipedia --module_type graph_attention --prior_ratio 0.5 --learning_rate 1e-3 

   

## Inference
Pruning the edge data in the database inductively according to the trained Graph pruning network above.    

    python edge_pruning.py --data_set wikipedia  --output_edge_txt ./result/edge_pred.txt --ckpt_file  ./lightning_logs_gsn/lightning_logs/version_0/checkpoints/epoch=10.ckpt

## Evaluation
Using a gnn to evaluate the performance of graph pruning.(this requires a trained gnn model from the supervised task, eg. runing the following commands on dynamic node classification).
    
    python train_gnn.py --mode origin --data_set wikipedia

    python eval_gnn.py --data_set wikipedia --mode gsn --pruning_ratio 0.5 --mask_edge --output_edge_txt ./result/edge_pred.txt --ckpt_file ./lightning_logs_gnn/lightning_logs/version_0/checkpoints/epoch=10.ckpt


# Cite
```bibtex
@article{li2022step,
  title   = {STEP: Self-supervised Temporal Graph Pruning at Scale},
  author  = {Jintang Li and Sheng Tian},
  year    = {2022}
}
```
