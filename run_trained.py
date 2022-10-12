# import time
# from datetime import datetime

# from utils.data_utils import load_dataset, make_dataloader, make_transforms, create_imbalance, split_dataset
# from utils.logger_utils import Logger
# from utils.test_utils import make_evaluate_fn, make_monitor_fn
# from utils.model_utils import make_model
# from utils.loss_utils import focal_loss
# from utils.mcr_utils import MaximalCodingRateReduction
# from core.fed_avg import FEDAVG
# from core.fed_pd import FEDPD
# from core.scaffold import SCAFFOLD
# # from core.mcr import MCR TODO decide if this is needed
# from config import make_parser
# from torch.utils.tensorboard import SummaryWriter
# import torch.nn.functional as F
# import warnings 

import torch
import os, json
from model import convnet, mlp, resnet, resnet_mcr



path = 'mcr/' + 'cifar100_dir_2022-0916-180629'
f = open(path+'/hparams.json')
data = json.load(f)

if data['loss_fn'] == 'mcr' and data['model']=='resnet':

    model = resnet_mcr.ResNet18(data['fd']).to(data['device'])

model.load_state_dict(torch.load(path+'/model.pth'))


