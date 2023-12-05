import time
import copy
import torch
import sys,os
# import torch.nn
from torch.utils.data import DataLoader
local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(local_dir)
from utils.train import Config, Trainer

config = Config()
trainer = Trainer(config)
trainer.run_trainer()
