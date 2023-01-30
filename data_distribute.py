import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings(action='ignore')

import sys
import argparse
import random
import importlib

import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler

sys.path.append('configs')
sys.path.append('module/data')
sys.path.append('module/model')
sys.path.append('module/trainer')

from dataset import BirdDataset
from basemodel import BirdModel
from trainer import Trainer

def get_config():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-C", "--config", help="config filename", default="cfg_baseline")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    cfg = importlib.import_module(args.config).cfg
    cfg.local_rank  = args.local_rank

    # ddp
    if cfg.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        cfg.device = torch.device("cuda", args.local_rank)
        cfg.rank = torch.distributed.get_rank()
        cfg.world_size = torch.distributed.get_world_size()  

    else:
        cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cfg.rank = 0

    return cfg

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataframe(cfg):
    name2dict = dict()
    df = pd.read_csv(cfg.train_file)
    df.rename(columns = {'labels' : 'labels_name'}, inplace = True)

    for label, names in zip(list(set(df.labels_name)), range(450)):
        name2dict[label] = names

    df["labels"] = df["labels_name"].apply(lambda x: name2dict[x])

    return df

def get_sampler(cfg, data_set, mode):
    if mode == "train":
        if cfg.local_rank != -1:
            sampler = DistributedSampler(data_set, drop_last=True)
        else:        
            sampler = RandomSampler(data_set)
    else:
        sampler = SequentialSampler(data_set)

    return sampler

def get_dataloader(cfg, df, mode):
    data_set = BirdDataset(df, cfg.transforms, mode=mode)
    sampler = get_sampler(cfg, data_set, mode)
    
    data_loader = DataLoader(
        data_set, 
        batch_size=cfg.batch_size, 
        sampler=sampler,
        shuffle=False,
        drop_last=True if mode == "trian" else False,
        num_workers=16, 
        pin_memory=True
    )

    return data_loader, sampler

def get_model(cfg):
    model = BirdModel(cfg)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(cfg.device)

    if cfg.local_rank != -1:
        model = DistributedDataParallel(model,
                                        device_ids=[cfg.local_rank],
                                        output_device=cfg.local_rank,
                                        find_unused_parameters=False)                                       
    else:
        model = torch.nn.DataParallel(model)
    return model

def main():
    cfg = get_config()
    seed_everything(42)
    df = get_dataframe(cfg)

    train_loader, train_sampler = get_dataloader(cfg, df, mode="train")
    valid_loader, _ = get_dataloader(cfg, df, mode="valid")
    test_loader, _ = get_dataloader(cfg, df, mode="test")

    model = get_model(cfg)
    trainer = Trainer(model, train_loader, valid_loader, test_loader, train_sampler, cfg)
    best_score = trainer.train()
    print(best_score)

if __name__ == "__main__":
    main()