import os
import gc
import torch
import argparse
import numpy as np
from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from data_loader import get_loader

def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True
        
def assign_gpu(gpu_ids):
    using_cuda = len(gpu_ids) > 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % int(gpu_ids[0]) if using_cuda else 'cpu')
    return device

if __name__ == '__main__':
    args = get_args()
    args.device = assign_gpu([0])
    dataset = str.lower(args.dataset.strip())

    print(args.batch_size)
    set_seed(args.seed)

    print("Start loading the data....")
    train_config = get_config(dataset, mode='train',
                              batch_size=args.batch_size)
    valid_config = get_config(dataset, mode='valid',
                              batch_size=args.batch_size)
    test_config = get_config(dataset, mode='test',  batch_size=args.batch_size)

    train_loader = get_loader(args, train_config, shuffle=True)

    print(args.dataset, "!!!!!!!!!")

    print('Training data loaded!')
    valid_loader = get_loader(args, valid_config, shuffle=False)
    print('Validation data loaded!')
    test_loader = get_loader(args, test_config, shuffle=False)
    print('Test data loaded!')
    print('Finish loading the data....')

    # addintional appending
    args.word2id = train_config.word2id

    # architecture parametersls
    #       768         20          5
    args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim
    args.dataset = args.data = dataset
    args.when = args.when
    args.n_class = output_dim_dict.get(dataset, 1)
    args.criterion = criterion_dict.get(dataset, 'MSELoss')

    solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)
    solver.train_and_eval()
