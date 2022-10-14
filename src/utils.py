import numpy as np


def get_neighbor_finder(data, uniform, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(adj_list, uniform=uniform)


def masked_get_neighbor_finder(data, edge_score_dict, pruning_ratio,uniform, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    count = 0
    score_list = list(edge_score_dict.values())
    c = int(len(score_list) * pruning_ratio)
    score_list.sort()
    threshold = score_list[c]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
        temp_score = edge_score_dict[edge_idx]
        if temp_score > threshold:
            count += 1
            adj_list[source].append((destination, edge_idx, timestamp))
            adj_list[destination].append((source, edge_idx, timestamp))

    print('Retain %f of edge, total %d edge' %(count/len(edge_score_dict), len(edge_score_dict)) )
    return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []
        self.node_to_edge_type = []

        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # We sort the list based on timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

        self.uniform = uniform

        if seed is not None:
          self.seed = seed
          self.random_state = np.random.RandomState(self.seed)


    def find_before(self, src_idx_list, cut_time_list, n_neighbors, exclude_node):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps
        """
        neighbor_nodes_array = np.array([], dtype=np.int64)
        node_to_edge_idxs_array = np.array([], dtype=np.int64)
        node_to_edge_timestamps_array = np.array([])
        edge_list = np.array([], dtype=np.int64).reshape([-1, 2])
        for idx, (src_idx, cut_time) in enumerate(zip(src_idx_list, cut_time_list)):
            i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
            neighbor_nodes = self.node_to_neighbors[src_idx][:i]
            neighbor_edge_idxs = self.node_to_edge_idxs[src_idx][:i]
            neighbor_times = self.node_to_edge_timestamps[src_idx][:i]

            index = np.where(~np.isin(neighbor_nodes, np.array(exclude_node)))
            neighbor_nodes = neighbor_nodes[index]
            neighbor_edge_idxs = neighbor_edge_idxs[index]
            neighbor_times = neighbor_times[index]

            n_min_neighbors = min(len(neighbor_nodes), n_neighbors)
            if n_min_neighbors > 0:
                if self.uniform:
                    sampled_idx = np.random.choice(range(len(neighbor_nodes)), n_min_neighbors, replace=False)
                    neighbor_nodes = neighbor_nodes[sampled_idx]
                    neighbor_edge_idxs = neighbor_edge_idxs[sampled_idx]
                    neighbor_times = neighbor_times[sampled_idx]
                else:
                    neighbor_nodes = neighbor_nodes[-n_min_neighbors:]
                    neighbor_edge_idxs = neighbor_edge_idxs[-n_min_neighbors:]
                    neighbor_times = neighbor_times[-n_min_neighbors:]


            temp_srcid = np.array([src_idx] * len(neighbor_nodes))
            temp_edge_id = np.column_stack((temp_srcid, neighbor_nodes))

            neighbor_nodes_array = np.concatenate((neighbor_nodes_array, neighbor_nodes))
            node_to_edge_idxs_array = np.concatenate((node_to_edge_idxs_array, neighbor_edge_idxs))
            node_to_edge_timestamps_array = np.concatenate((node_to_edge_timestamps_array, neighbor_times))
            edge_list = np.concatenate((edge_list, temp_edge_id.astype(np.int64)))

        return neighbor_nodes_array, node_to_edge_idxs_array, node_to_edge_timestamps_array, edge_list


    def get_temporal_neighbor_all(self, source_node, timestamp, n_layer, n_neighbors):
        #assert (len(source_node) == len(timestamp))

        edge_list = np.array([], dtype=np.int64).reshape([-1, 2])
        time_list = np.array([])
        idx_list = np.array([], dtype=np.int64)

        temp_center_node = [source_node]
        temp_center_time = [timestamp]
        exclude_node = []
        for i in range(n_layer):

            neighbor_nodes, neighbor_edge_idxs, neighbor_times, neighbor_edge_node= self.find_before(temp_center_node, temp_center_time, n_neighbors, exclude_node)
            if len(neighbor_nodes) > 0 and n_neighbors>0:
                idx_list = np.concatenate((idx_list, neighbor_edge_idxs))
                time_list = np.concatenate((time_list, neighbor_times))
                edge_list = np.concatenate((edge_list, neighbor_edge_node))
                exclude_node = temp_center_node
                temp_center_node = np.unique(neighbor_nodes).tolist()
                temp_center_time = [timestamp] * len(temp_center_node)
            else:
                break



        return edge_list, time_list, idx_list



class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)
