import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as scatter
from modules.utils import MergeLayer_output, Feat_Process_Layer
from modules.embedding_module import get_embedding_module
from modules.time_encoding import TimeEncode
from model.gsn import Graph_sampling_network
from model.gpn import Graph_pruning_network


class TGAT(torch.nn.Module):
    def __init__(self, config, embedding_module_type="graph_attention"):
        super().__init__()
        self.cfg = config

        self.nodes_dim = self.cfg.input_dim
        self.edge_dim = self.cfg.input_dim
        self.dims = self.cfg.hidden_dim

        self.n_heads = self.cfg.n_heads
        self.dropout = self.cfg.drop_out
        self.n_layers = self.cfg.n_layer

        self.mode = self.cfg.mode

        self.time_encoder = TimeEncode(dimension=self.dims)
        self.embedding_module_type = embedding_module_type
        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     node_features_dims=self.dims,
                                                     edge_features_dims=self.dims,
                                                     time_features_dim=self.dims,
                                                     hidden_dim=self.dims,
                                                     n_heads=self.n_heads, dropout=self.dropout)

        self.node_preocess_fn = Feat_Process_Layer(self.nodes_dim, self.dims)
        self.edge_preocess_fn = Feat_Process_Layer(self.edge_dim, self.dims)
        self.affinity_score = MergeLayer_output(self.dims, self.dims, drop_out=0.2)

        self.predictor = nn.Sequential(nn.Linear(self.dims, self.dims)) # output layer

        self.gsn = Graph_sampling_network(self.dims, self.cfg.batch_size, mask_ratio=self.cfg.prior_ratio)
        self.edge_precom = Graph_pruning_network(self.edge_dim, self.dims, self.dropout)

    def forward(self, src_org_edge_feat, src_edge_to_time, src_center_node_idx, src_neigh_edge, src_node_features):
        # apply tgat
        source_node_embedding, src_edge_feat = self.compute_temporal_embeddings(src_neigh_edge, src_edge_to_time,
                                                                                src_org_edge_feat, src_node_features)

        loclsrc_node_embedding = source_node_embedding[src_center_node_idx, :]
        score = self.affinity_score(loclsrc_node_embedding, loclsrc_node_embedding)
        return score


    def forward_gsn(self, src_org_edge_feat, src_edge_to_time, src_center_node_idx, src_neigh_edge, src_node_features,
                init_edge_index, batch_idx, step=0):

        # apply tgat
        source_node_embedding, src_edge_feat = self.compute_temporal_embeddings(src_neigh_edge, src_edge_to_time,
                                                                src_org_edge_feat, src_node_features)

        loclsrc_node_embedding = source_node_embedding[src_center_node_idx,:]

        source_node_embedding_clone = source_node_embedding
        src_edge_feat_clone = src_edge_feat
        time_encodding = self.time_encoder(src_edge_to_time)
        src_edge_probs, src_edge_mask = self.gsn.forward(source_node_embedding_clone, src_neigh_edge, time_encodding,
                                                 src_edge_feat_clone, batch_idx, src_center_node_idx)

        gsn_node_embedding, _ = self.compute_temporal_embeddings(src_neigh_edge, src_edge_to_time,
                                                             src_org_edge_feat, src_node_features,
                                                             None, src_edge_probs)
        gsnsrc_node_embedding = gsn_node_embedding[src_center_node_idx, :]


        unique_edge_label = self.Merge_same_edge(init_edge_index, src_edge_mask)
        temp_edge_label = unique_edge_label.long()
        edge_logit = self.edge_precom(src_org_edge_feat)
        loss_edge_pred = self.edge_precom.loss(edge_logit.reshape([-1, 2]), temp_edge_label)

        loss_sparse = self.gsn.sparse_loss(src_edge_probs)
        loss_mi = self.ddgcl(loclsrc_node_embedding, gsnsrc_node_embedding)
        max_probs = torch.max(src_edge_probs)
        min_probs = torch.min(src_edge_probs)
        return {'loss': loss_mi, 'loss_sparse': loss_sparse, 'loss_edge_pred':loss_edge_pred,
                'edge_index': src_neigh_edge, 'edge_probs': src_edge_probs,
                'max_probs':max_probs, 'min_probs':min_probs}


    def compute_temporal_embeddings(self, neigh_edge, edge_to_time, edge_feat, node_feat, edge_mask=None, sample_ratio=None):
        node_feat = self.node_preocess_fn(node_feat)
        edge_feat = self.edge_preocess_fn(edge_feat)

        node_embedding = self.embedding_module.compute_embedding(neigh_edge, edge_to_time,
                                                                 edge_feat, node_feat, edge_mask, sample_ratio)

        return node_embedding, edge_feat

    def ddgcl(self, x1, x2):
        x1 = self.predictor(x1)
        l_pos = torch.sigmoid(torch.sum(x1 * x2, dim=-1)).reshape([-1, 1])
        l_neg = torch.sigmoid(torch.sum(torch.einsum('nc,kc->nkc', x1, x2), dim=-1))
        matrix = torch.diag_embed(torch.diag(l_neg))
        l_neg = l_neg - matrix

        label1 = torch.ones_like(l_pos)
        label2 = torch.zeros_like(l_neg)
        logits = torch.cat([l_pos, l_neg], dim=1).reshape([-1])
        labels = torch.cat([label1, label2], dim=1).reshape([-1])
        loss_bce = torch.nn.BCELoss()
        loss = loss_bce(logits, labels)
        return loss

    def Merge_same_edge(self, init_edge_index, src_edge_mask):
        output, _ = scatter.scatter_max(src_edge_mask, init_edge_index, dim=0)
        output = output[init_edge_index]
        return output