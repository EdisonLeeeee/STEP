import torch
from modules.utils import MergeLayer_output, Feat_Process_Layer


class Graph_pruning_network(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_out):
        super(Graph_pruning_network, self).__init__()
        self.edge_dim = input_dim
        self.dims = hidden_dim
        self.dropout = drop_out

        self.affinity_score = Precomput_output(self.edge_dim, self.dims, 2, drop_out=0.2)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, edge_feat):

        edge_logit = self.affinity_score(edge_feat)

        return edge_logit

class Precomput_output(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3=2, drop_out=0.2):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1, dim2)
        self.fc2 = torch.nn.Linear(dim2, dim2)
        self.fc3 = torch.nn.Linear(dim2, dim3)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop_out)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.dropout(self.act(self.fc2(h)))
        h = self.fc3(h)
        h = self.concrete_sample(h)
        return h

    def concrete_sample(self, log_alpha, beta=1.0):
        if self.training:
            log_alpha = log_alpha.reshape([-1])
            bias = 0.1
            random_noise = torch.empty(log_alpha.shape, dtype = log_alpha.dtype, device=log_alpha.device).uniform_(bias, 1-bias)
            gate_inputs = torch.log(random_noise) - torch.log(1-random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = gate_inputs.reshape([-1, 2])
        else:
            gate_inputs = log_alpha
        return gate_inputs
