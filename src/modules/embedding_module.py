import torch
from torch import nn
import numpy as np
import math

from modules.temporal_attention import TemporalAttentionLayer2

class EmbeddingModule(nn.Module):
  def __init__(self, time_encoder, n_layers,
               node_features_dims, edge_features_dims, time_features_dim, hidden_dim, dropout):
    super(EmbeddingModule, self).__init__()
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = node_features_dims
    self.n_edge_features = edge_features_dims
    self.n_time_features = time_features_dim
    self.dropout = dropout
    self.embedding_dimension = hidden_dim

  def compute_embedding(self, neigh_edge, edge_to_time, edge_feat, node_feat):
    pass



class GraphEmbedding(EmbeddingModule):
  def __init__(self, time_encoder, n_layers,
               node_features_dims, edge_features_dims, time_features_dim, hidden_dim, n_heads=2, dropout=0.1):
    super(GraphEmbedding, self).__init__(time_encoder, n_layers,
                                         node_features_dims, edge_features_dims, time_features_dim,
                                         hidden_dim, dropout)


  def compute_embedding(self, neigh_edge, edge_to_time, edge_feat, node_feat, edge_mask=None, sample_ratio=None):
    '''
    :param neigh_edge:  [E, 2]
    :param edge_to_time:  [E]
    :param edge_feat:  [E, D]
    :param node_feat: [N, D]
    :return:
    '''

    n_layers = self.n_layers
    assert (n_layers >= 0)

    temp_node_feat = node_feat
    src_time_embeddings = self.time_encoder(torch.zeros_like(edge_to_time))
    edge_time_embeddings = self.time_encoder(edge_to_time)

    mask = edge_mask
    for layer in range(n_layers):

        temp_node_feat = self.aggregate(n_layers, temp_node_feat,
                                        neigh_edge,
                                        edge_feat,
                                        src_time_embeddings,
                                        edge_time_embeddings,
                                        mask, sample_ratio)

    out = temp_node_feat
    return out

  def aggregate(self, n_layers, node_features, edge_index,
                edge_feature,
                src_time_features, edge_time_embeddings, mask, sample_ratio):
    return None


# class GraphSumEmbedding(GraphEmbedding):
#   def __init__(self, time_encoder, n_layers, node_features_dims, edge_features_dims,
#                time_features_dim, hidden_dim, n_heads=2, dropout=0.1):
#     super(GraphSumEmbedding, self).__init__( time_encoder=time_encoder, n_layers=n_layers,
#                                             node_features_dims=node_features_dims,
#                                             edge_features_dims=edge_features_dims,
#                                             time_features_dim=time_features_dim,
#                                             hidden_dim=hidden_dim,
#                                             n_heads=n_heads, dropout=dropout)
#
#     self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(hidden_dim + time_features_dim +
#                                                          edge_features_dims, hidden_dim)
#                                          for _ in range(n_layers)])
#     self.linear_2 = torch.nn.ModuleList(
#       [torch.nn.Linear(hidden_dim + node_features_dims + time_features_dim,
#                        hidden_dim) for _ in range(n_layers)])
#
#   def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
#                 neighbor_embeddings,
#                 edge_time_embeddings, edge_features, mask):
#     neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
#                                    dim=2)
#     neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
#     neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))
#
#     source_features = torch.cat([source_node_features,
#                                  source_nodes_time_embedding.squeeze()], dim=1)
#     source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
#     source_embedding = self.linear_2[n_layer - 1](source_embedding)
#
#     return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, time_encoder, n_layers, node_features_dims, edge_features_dims,
               time_features_dim, hidden_dim, n_heads=2, dropout=0.1):
    super(GraphAttentionEmbedding, self).__init__(time_encoder, n_layers,
                                                  node_features_dims, edge_features_dims,
                                                  time_features_dim,
                                                  hidden_dim,
                                                  n_heads, dropout)

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer2(
      n_node_features=node_features_dims,
      n_neighbors_features=node_features_dims,
      n_edge_features=edge_features_dims,
      time_dim=time_features_dim,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=hidden_dim)
      for _ in range(n_layers)])

  def aggregate(self, n_layer, node_features, edge_index,
                edge_feature,
                src_time_features, edge_time_embeddings, mask, sample_ratio):
    attention_model = self.attention_models[n_layer - 1]

    source_embedding = attention_model(node_features,
                                          edge_index,
                                          edge_feature,
                                          src_time_features,
                                          edge_time_embeddings,
                                          mask, sample_ratio)

    return source_embedding



def get_embedding_module(module_type, time_encoder, n_layers,
                         node_features_dims, edge_features_dims, time_features_dim,
                         hidden_dim, n_heads=2, dropout=0.1):

  if module_type == "graph_attention":
    return GraphAttentionEmbedding( time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    node_features_dims=node_features_dims,
                                    edge_features_dims=edge_features_dims,
                                    time_features_dim=time_features_dim,
                                    hidden_dim=hidden_dim,
                                    n_heads=n_heads, dropout=dropout)
  # elif module_type == "graph_sum":
  #   return GraphSumEmbedding(time_encoder=time_encoder,
  #                             n_layers=n_layers,
  #                            node_features_dims=node_features_dims,
  #                            edge_features_dims=edge_features_dims,
  #                            time_features_dim=time_features_dim,
  #                            hidden_dim=hidden_dim,
  #                            n_heads=n_heads, dropout=dropout)

  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))


