import argparse


parser = argparse.ArgumentParser(description='Denoise')

parser.add_argument('--dir_data', type=str, default='../dataset')
parser.add_argument('--data_set', type=str, default='wikipedia')
parser.add_argument('--output_edge_txt', type=str, default='./result/edge_pred.txt')
parser.add_argument('--mask_edge', action='store_true', default=False)
parser.add_argument('--bipartite', action='store_true')
parser.add_argument('--mode', type=str, default='gsn', choices=('origin', 'dropedge', 'gsn'))
parser.add_argument('--prior_ratio', type=float, default=0.5)
parser.add_argument('--pruning_ratio', type=float, default=0.5)

##data param
parser.add_argument('--n_neighbors', type=int, default=20, help='Maximum number of connected edge per node')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_data_workers', type=int, default=15)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--accelerator', type=str, default='dp')

##model param
parser.add_argument('--ckpt_file', type=str, default='./')
parser.add_argument('--input_dim', type=int, default=172)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--learning_rate', type=float, default=5e-4)

args = parser.parse_args()
