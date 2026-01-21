import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn

# change to your own path
MODEL_PATH = '../BERT/bert-base-uncased'
MOSI_PKL_PATH = '../datasets/MOSI/mosi_data_noalign.pkl'
MOSI_CSV_PATH = '../datasets/MOSI/MOSI-label.csv'
MOSEI_PKL_PATH = '../datasets/MOSEI/mosi_data_noalign.pkl'
MOSEI_CSV_PATH = '../datasets/MOSEI/MOSI-label.csv'


username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
print(sdk_dir)
data_dir = Path('/root/datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
}

criterion_dict = {
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
}


def get_args():
    parser = argparse.ArgumentParser(
        description='MOSI-and-MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)


    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei'],
                        help='dataset to use (default: mosei)')
    parser.add_argument('--data_path', type=str, default='datasets',
                        help='path for storing the dataset')
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')
    parser.add_argument('--d_prjh', type=int, default=128,
                        help='hidden size in projection network')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',  
                        help='batch size (default: 32)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr_main', type=float, default=1e-4,
                        help='initial learning rate for main model parameters (default: 1e-3)')
    parser.add_argument('--lr_bert', type=float, default=5e-5,
                        help='initial learning rate for bert parameters (default: 5e-5)')
    parser.add_argument('--alpha', type=float, default=0.35,
                        help='weight for loss1')
    parser.add_argument('--beta', type=float, default=0.25,
                        help='weight for loss2')

    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_bert', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')

    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--update_batch', type=int, default=1,
                        help='update batch interval')


    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                        help='attention dropout (for audio)')
    parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                        help='attention dropout (for visual)')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0,
                        help='output layer dropout')

    parser.add_argument('--num_heads', type=int, default=4,
                        help='number of heads for the transformer network (default: 4)')
    parser.add_argument('--layers', type=int, default=3,
                        help='number of layers in the network (default: 3)')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='embed_dim')

    parser.add_argument('--attn_mask', default=True,
                        help='use attention mask for Transformer (default: true)')
    
    parser.add_argument('--model_name', default='none',
                        help='model_name')
    
    parser.add_argument('--add2logpath', type=str, default='none',
                        help='add2logpath')
    
    args = parser.parse_args()
    return args


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, data, mode='train'):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.dataset_dir = data_dict[data.lower()]
        self.sdk_dir = sdk_dir
        self.mode = mode

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(dataset='mosi', mode='train', batch_size=32):
    config = Config(data=dataset, mode=mode)

    config.dataset = dataset
    config.batch_size = batch_size

    return config
