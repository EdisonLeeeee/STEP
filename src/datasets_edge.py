import torch
import torch.utils.data
import os
import numpy as np
import random
import pandas as pd


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)



class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config

        dataset_name = '{}/ml_{}'.format(self.config.dir_data, self.config.data_set)

        self.full_data, self.positive_eids, self.edge_features, self.node_features = \
            self.get_data(dataset_name)

        self.index_start = self.positive_eids[0]


    def get_data(self, dataset_name):
        graph_df = pd.read_csv('{}.csv'.format(dataset_name))
        edge_features = np.load('{}.npy'.format(dataset_name))
        node_features = np.load('{}_node.npy'.format(dataset_name))


        sources = graph_df.u.values
        destinations = graph_df.i.values
        edge_idxs = graph_df.idx.values
        labels = graph_df.label.values
        timestamps = graph_df.ts.values

        random.seed(2020)

        positive_eids = np.where(timestamps >= 0)[0]


        full_data = Data(sources, destinations, timestamps, edge_idxs, labels)


        return full_data, positive_eids, edge_features, node_features


    def __getitem__(self, item):
        item += self.index_start

        edge_idx = self.full_data.edge_idxs[item]

        edge_feature = self.edge_features[edge_idx]
        edge_idx = np.array(edge_idx)

        return {
            'edge_feature': torch.from_numpy(edge_feature.astype(np.float32)).reshape(1,-1),
            'edge_idx': torch.from_numpy(edge_idx).reshape(1)
        }


    def __len__(self):
        return len(self.positive_eids)

class Collate:
    def __init__(self, config):
        self.config = config

    def dyg_collate_fn(self, batch):
        edge_feature = torch.cat([b['edge_feature'] for b in batch], dim=0)   #n1,f
        edge_idx = torch.cat([b['edge_idx'] for b in batch], dim=0)  # n

        return {
            'edge_feature':edge_feature,
            'edge_idx': edge_idx
        }


class RandomDropSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, dataset, drop_rate):
        self.dataset = dataset
        self.drop_rate = drop_rate
        self.drop_num = int(len(dataset) * drop_rate)

    def __iter__(self):
        arange = np.arange(len(self.dataset))
        np.random.shuffle(arange)
        #indices = arange[: (1 - self.drop_num)]
        #return iter(np.sort(indices))
        indices = arange
        return iter(indices)

    def __len__(self):
        return len(self.dataset) - self.drop_num


