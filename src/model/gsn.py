import torch
import torch.nn.functional as F
import torch_scatter as scatter


class Graph_sampling_network(torch.nn.Module):
    def __init__(self, dim, batch_size, mask_ratio=0.5):
        super(Graph_sampling_network, self).__init__()
        self.mask_act = 'sigmoid'
        self.mask_ratio = mask_ratio
        self.dim = dim
        self.batch_size = batch_size

        self.elayers1 = torch.nn.Sequential(
            torch.nn.Linear(self.dim * 4, self.dim),
            torch.nn.ReLU()
        )

        self.elayers3 = torch.nn.Sequential(
            torch.nn.Linear(2 + self.dim, 1)
            #torch.nn.Linear(2, 1)
        )
#        torch.nn.init.xavier_normal_(self.elayers2.weight)

    def concrete_sample(self, log_alpha, beta=1.0):
        if self.training:
            bias = 0.1
            random_noise = torch.empty(log_alpha.shape, dtype=log_alpha.dtype, device=log_alpha.device).uniform_(bias, 1-bias)
            gate_inputs = torch.log(random_noise) - torch.log(1-random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        return gate_inputs


    def forward(self, node_embeddings, edge_index, time_encodding, edge_feat, batch_idx, src_center_node_idx):
        node_i = edge_index[:, 0]
        node_j = edge_index[:, 1]

        node_feat_i = node_embeddings[node_i, :]
        node_feat_j = node_embeddings[node_j, :]
        center_node_feat = node_embeddings[src_center_node_idx, :]

        h = torch.cat([node_feat_i, node_feat_j, edge_feat, time_encodding], dim=1)
        h1 = self.elayers1(h)

        redundancy_score = self.redundancy_attention(h1)     #[n, 1]
        relevance_score = self.relevance_attention(h1, batch_idx.long(), center_node_feat) #[n, 1]

        attn_score = torch.cat([redundancy_score, relevance_score, h1], dim=-1)

        log_alpha = self.elayers3(attn_score)
        edge_sample_probs = self.concrete_sample(log_alpha)
        edge_sample_probs = edge_sample_probs.reshape([-1])

        _, rank_idx = edge_sample_probs.sort(dim=0)
        cut_off_nums = round(edge_sample_probs.shape[0] * self.mask_ratio)
        low_idx = rank_idx[:cut_off_nums]
        high_idx = rank_idx[cut_off_nums:]


        edge_mask = edge_sample_probs.clone().detach()
        edge_mask[low_idx] = 0
        edge_mask[high_idx] = 1

        return edge_sample_probs, edge_mask.byte()

    def redundancy_attention(self, x):
        x = F.normalize(x, p=2, dim=1)
        dots = x @ x.transpose(-1, -2) #[m, m]
        attn = torch.softmax(dots, dim=-1)
        out = attn - torch.diag_embed(torch.diag(attn))
        out = torch.sum(out, dim=-1)
        return out.reshape([-1, 1])

    def relevance_attention(self, x, batch_id, center_node_feat):
        all_node_feat = center_node_feat[batch_id-1, :]

        dots = torch.sum(torch.multiply(x, all_node_feat), dim=-1 )
        attn = scatter.composite.scatter_softmax(dots, batch_id)

        return attn.reshape([-1, 1])


    def drop_edge(self, edge_index, batch_idx):
        edge_sample_probs = torch.rand(edge_index.shape[0])

        # y = torch.unique(batch_idx)
        # mask_idx = [ self.get_mask_by_batch_fn(edge_sample_probs, batch_idx, x) for x in y]
        # low_idx = torch.cat([x[0] for x in mask_idx], dim=0)
        # high_idx= torch.cat([x[1] for x in mask_idx], dim=0)

        _, rank_idx = edge_sample_probs.sort(dim=0)
        cut_off_nums = round(edge_sample_probs.shape[0] * self.mask_ratio)
        low_idx = rank_idx[:cut_off_nums]
        high_idx = rank_idx[cut_off_nums:]



        edge_mask = edge_sample_probs.clone()
        edge_mask[low_idx] = 0
        edge_mask[high_idx] = 1

        return edge_mask.byte()

    def get_mask_by_batch_fn(self, edge_sample_probs, batch_idx, x):
        index = torch.nonzero(torch.where(batch_idx == x, batch_idx.clone().detach(), torch.tensor(0.0, device=x.device))).reshape([-1])
        edge_sample_probs = edge_sample_probs[index]
        _, rank_idx = edge_sample_probs.sort(dim=0)
        cut_off_nums = round(edge_sample_probs.shape[0] * self.mask_ratio)
        low_idx = rank_idx[:cut_off_nums]
        true_low_idx = index[low_idx]
        high_idx = rank_idx[cut_off_nums:]
        true_high_idx = index[high_idx]
        return true_low_idx, true_high_idx

    def sparse_loss(self, log_alpha):
        var_x = torch.mean(log_alpha * log_alpha) - torch.mean(log_alpha) * torch.mean(log_alpha)
        loss_1 = torch.abs(var_x - self.mask_ratio * (1 - self.mask_ratio))
        loss_2 = torch.abs(torch.mean(log_alpha) - (1 - self.mask_ratio))
        loss = 1 * loss_1 + 1 * loss_2
        return loss