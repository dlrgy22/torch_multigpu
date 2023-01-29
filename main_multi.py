import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
from basemodel_multigpu import BirdModelMultiGPU
from trainer_multigpu import MultiGPUTrainer

def get_config():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-C", "--config", help="config filename", default="cfg_baseline")
    args = parser.parse_args()
    cfg = importlib.import_module(args.config).cfg

    cfg.device = ["cuda:0", "cuda:1"]

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
        sampler = RandomSampler(data_set)
    else:
        sampler = SequentialSampler(data_set)

    return sampler

def get_dataloader(cfg, df, mode):
    data_set = BirdDataset(df, cfg.transforms, mode=mode)
    sampler = get_sampler(cfg, data_set, mode)
    
    data_loader = DataLoader(data_set, batch_size=cfg.batch_size, sampler=sampler, num_workers=16, pin_memory=True)
    return data_loader, sampler

def get_model(cfg):
    model = BirdModelMultiGPU(cfg)
    model.to_gpus()
    
    return model

def main():
    cfg = get_config()
    seed_everything(42)
    df = get_dataframe(cfg)

    train_loader, train_sampler = get_dataloader(cfg, df, mode="train")
    valid_loader, _ = get_dataloader(cfg, df, mode="valid")
    test_loader, _ = get_dataloader(cfg, df, mode="test")

    model = get_model(cfg)
    trainer = MultiGPUTrainer(model, train_loader, valid_loader, test_loader, train_sampler, cfg)
    best_score = trainer.train()
    print(best_score)

if __name__ == "__main__":
    main()